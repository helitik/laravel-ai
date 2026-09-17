<?php

namespace Laravel\Ai\Gateway\OpenAiCompatible;

use Generator;
use Illuminate\Support\Str;
use Laravel\Ai\Streaming\Events\ReasoningDelta;
use Laravel\Ai\Streaming\Events\ReasoningEnd;
use Laravel\Ai\Streaming\Events\ReasoningStart;
use Laravel\Ai\Streaming\Events\StreamEvent;

/**
 * Tracks the reasoning block of a Chat Completions response.
 */
class ChatCompletionReasoning
{
    /**
     * The provider content block key holding the readable reasoning text, which doubles as DeepSeek's inbound field name.
     */
    public const CONTENT_BLOCK_KEY = 'reasoning_content';

    /**
     * The provider content block key holding the structured reasoning details.
     */
    public const DETAILS_BLOCK_KEY = 'reasoning_details';

    /**
     * The provider content block key recording which wire field the readable text arrived under.
     */
    public const FIELD_BLOCK_KEY = 'reasoning_field';

    /**
     * The wire fields carrying plain text reasoning, in the order they are read.
     */
    public const TEXT_FIELDS = ['reasoning', self::CONTENT_BLOCK_KEY];

    /**
     * The ID shared by the events of the currently open reasoning block.
     */
    protected string $reasoningId = '';

    /**
     * Indicates if a reasoning block is currently open.
     */
    protected bool $open = false;

    /**
     * The reasoning text accumulated so far.
     */
    protected string $text = '';

    /**
     * The structured reasoning details accumulated so far, keyed by block identity.
     *
     * @var array<array-key, array<string, mixed>>
     */
    protected array $details = [];

    /**
     * The number of details seen that carried no id or index to merge on.
     */
    protected int $unkeyed = 0;

    /**
     * The wire field the readable reasoning text arrived under.
     */
    protected ?string $field = null;

    /**
     * Indicates if the stream carried a reasoning details array, including an empty one.
     */
    protected bool $sawDetails = false;

    /**
     * Where this stream reads its reasoning text from, either "plain" or "details".
     */
    protected ?string $source = null;

    public function __construct(protected string $invocationId) {}

    /**
     * Emit the reasoning events for the given streaming delta.
     *
     * @param  array<string, mixed>  $delta
     * @return Generator<int, StreamEvent>
     */
    public function process(array $delta): Generator
    {
        [$plain, $field] = static::plainFrom($delta);

        $this->field ??= $field;

        // Details are merged even when the text is read elsewhere, since only they can be replayed verbatim...
        $details = $this->mergeDetails($delta);

        // The first reasoning-bearing chunk picks the source, so mirrored fields are never counted twice...
        $this->source ??= match (true) {
            $plain !== '' => 'plain',
            $details !== '' => 'details',
            default => null,
        };

        $reasoning = match ($this->source) {
            'plain' => $plain,
            'details' => $details,
            default => '',
        };

        if ($reasoning !== '') {
            if (! $this->open) {
                $this->open = true;
                $this->reasoningId = $this->generateEventId();

                yield $this->event(new ReasoningStart($this->generateEventId(), $this->reasoningId, time()));
            }

            $this->text .= $reasoning;

            yield $this->event(new ReasoningDelta($this->generateEventId(), $this->reasoningId, $reasoning, time()));
        }

        if ($this->startsAnswer($delta)) {
            yield from $this->close();
        }
    }

    /**
     * Emit the reasoning end event if a block is still open.
     *
     * @return Generator<int, StreamEvent>
     */
    public function close(): Generator
    {
        if (! $this->open) {
            return;
        }

        $this->open = false;

        yield $this->event(new ReasoningEnd($this->generateEventId(), $this->reasoningId, time()));

        $this->reasoningId = '';
    }

    /**
     * Get the provider content blocks capturing the reasoning seen so far.
     *
     * @return array<string, mixed>
     */
    public function providerContentBlocks(): array
    {
        return static::providerContentBlocksFor(
            $this->text,
            $this->sawDetails ? array_values($this->details) : null,
            $this->field,
        );
    }

    /**
     * Get the provider content blocks capturing the reasoning of the given Chat Completions message.
     *
     * @param  array<string, mixed>  $message
     * @return array<string, mixed>
     */
    public static function providerContentBlocksIn(array $message): array
    {
        return static::providerContentBlocksFor(
            static::textFrom($message),
            static::detailsIn($message),
            static::plainFrom($message)[1],
        );
    }

    /**
     * Get the provider content blocks holding the given reasoning text and details.
     *
     * A null details array means the payload carried none, which an empty one does not.
     *
     * @param  array<int, array<string, mixed>>|null  $details
     * @return array<string, mixed>
     */
    public static function providerContentBlocksFor(string $reasoning, ?array $details = null, ?string $field = null): array
    {
        return array_filter([
            static::CONTENT_BLOCK_KEY => $reasoning === '' ? null : $reasoning,
            static::DETAILS_BLOCK_KEY => $details,
            static::FIELD_BLOCK_KEY => $reasoning === '' ? null : $field,
        ], fn (string|array|null $value): bool => $value !== null);
    }

    /**
     * Get the readable reasoning text to replay for the given assistant message provider content blocks.
     *
     * @param  array<array-key, mixed>  $providerContentBlocks
     */
    public static function replayableFrom(array $providerContentBlocks): ?string
    {
        $reasoning = $providerContentBlocks[static::CONTENT_BLOCK_KEY] ?? null;

        return is_string($reasoning) && $reasoning !== '' ? $reasoning : null;
    }

    /**
     * Get the wire field the readable reasoning text should be replayed under.
     *
     * @param  array<array-key, mixed>  $providerContentBlocks
     */
    public static function replayableFieldFrom(array $providerContentBlocks, string $default): string
    {
        $field = $providerContentBlocks[static::FIELD_BLOCK_KEY] ?? null;

        return in_array($field, static::TEXT_FIELDS, true) ? $field : $default;
    }

    /**
     * Get the structured reasoning details to replay verbatim, which carry signatures and encrypted payloads the readable text loses.
     *
     * @param  array<array-key, mixed>  $providerContentBlocks
     * @return array<int, array<string, mixed>>|null
     */
    public static function replayableDetailsFrom(array $providerContentBlocks): ?array
    {
        $details = $providerContentBlocks[static::DETAILS_BLOCK_KEY] ?? null;

        if (! is_array($details)) {
            return null;
        }

        $details = array_values(array_filter($details, is_array(...)));

        // OpenRouter requires the whole sequence back unmodified, so one unsigned block invalidates all of them...
        foreach ($details as $detail) {
            if (static::isUnsignedThinkingText($detail)) {
                return null;
            }
        }

        return $details;
    }

    /**
     * Determine if the given detail is a reasoning text block that lost the signature its format requires back.
     *
     * @param  array<string, mixed>  $detail
     */
    protected static function isUnsignedThinkingText(array $detail): bool
    {
        // Anthropic rejects unsigned thinking, and Gemini reports a corrupted thought signature...
        $signed = ['anthropic', 'google-gemini'];

        return ($detail['type'] ?? null) === 'reasoning.text'
            && Str::startsWith((string) ($detail['format'] ?? ''), $signed)
            && blank($detail['signature'] ?? null);
    }

    /**
     * Extract the reasoning text from a Chat Completions message or streaming delta.
     *
     * @param  array<string, mixed>  $payload
     */
    public static function textFrom(array $payload): string
    {
        $reasoning = static::plainFrom($payload)[0];

        return $reasoning === '' ? static::detailsTextFrom($payload) : $reasoning;
    }

    /**
     * Extract the structured reasoning details from a Chat Completions message or streaming delta.
     *
     * @param  array<string, mixed>  $payload
     * @return array<int, array<string, mixed>>
     */
    public static function detailsFrom(array $payload): array
    {
        return static::detailsIn($payload) ?? [];
    }

    /**
     * Extract the structured reasoning details, returning null when the payload carried none at all.
     *
     * @param  array<string, mixed>  $payload
     * @return array<int, array<string, mixed>>|null
     */
    public static function detailsIn(array $payload): ?array
    {
        $details = $payload[static::DETAILS_BLOCK_KEY] ?? null;

        return is_array($details) ? array_values(array_filter($details, is_array(...))) : null;
    }

    /**
     * Merge the given delta's reasoning details into the captured blocks, returning the text that newly arrived.
     *
     * @param  array<string, mixed>  $delta
     */
    protected function mergeDetails(array $delta): string
    {
        $arrived = '';

        $this->sawDetails = $this->sawDetails || static::detailsIn($delta) !== null;

        foreach (static::detailsFrom($delta) as $detail) {
            // Details of different types may share an id or index, and those carrying neither cannot be merged at all...
            $key = ($detail['type'] ?? '').'#'.($detail['id'] ?? $detail['index'] ?? 'unkeyed-'.$this->unkeyed++);

            if (! isset($this->details[$key])) {
                $this->details[$key] = $detail;

                $arrived .= static::readableTextFrom($detail);

                continue;
            }

            $arrived .= $this->mergeDetail($key, $detail);
        }

        return $arrived;
    }

    /**
     * Merge a repeated detail into the block already captured under the given key, returning the text that newly arrived.
     *
     * @param  array<string, mixed>  $detail
     */
    protected function mergeDetail(string $key, array $detail): string
    {
        $captured = static::readableTextFrom($this->details[$key]);

        // Repeated blocks stream incremental fragments that concatenate in order, per OpenRouter's reasoning contract...
        $arrived = static::readableTextFrom($detail);

        $this->details[$key] = [...$this->details[$key], ...$detail];

        foreach (['text', 'summary'] as $field) {
            if (array_key_exists($field, $this->details[$key])) {
                $this->details[$key][$field] = $captured.$arrived;

                break;
            }
        }

        return $arrived;
    }

    /**
     * Extract the plain text reasoning and the wire field it arrived under.
     *
     * @param  array<string, mixed>  $payload
     * @return array{0: string, 1: string|null}
     */
    protected static function plainFrom(array $payload): array
    {
        // OpenRouter and current vLLM send `reasoning`, while DeepSeek and older vLLM send `reasoning_content`...
        foreach (static::TEXT_FIELDS as $field) {
            $reasoning = $payload[$field] ?? null;

            if (is_string($reasoning) && $reasoning !== '') {
                return [$reasoning, $field];
            }
        }

        return ['', null];
    }

    /**
     * Extract the reasoning text carried by structured reasoning details.
     *
     * @param  array<string, mixed>  $payload
     */
    protected static function detailsTextFrom(array $payload): string
    {
        $text = '';

        foreach (static::detailsFrom($payload) as $detail) {
            $text .= static::readableTextFrom($detail);
        }

        return $text;
    }

    /**
     * Get the readable text of a single reasoning detail, which is empty for encrypted blocks.
     *
     * @param  array<string, mixed>  $detail
     */
    protected static function readableTextFrom(array $detail): string
    {
        $text = $detail['text'] ?? $detail['summary'] ?? null;

        return is_string($text) ? $text : '';
    }

    /**
     * Determine if the given streaming delta begins the model's answer.
     *
     * @param  array<string, mixed>  $delta
     */
    protected function startsAnswer(array $delta): bool
    {
        $content = $delta['content'] ?? null;

        return (is_string($content) && $content !== '') || filled($delta['tool_calls'] ?? null);
    }

    /**
     * Tag the given event with the invocation it belongs to.
     */
    protected function event(StreamEvent $event): StreamEvent
    {
        return $event->withInvocationId($this->invocationId);
    }

    /**
     * Generate a lowercase UUID v7 for use as a stream event ID.
     */
    protected function generateEventId(): string
    {
        return strtolower((string) Str::uuid7());
    }
}
