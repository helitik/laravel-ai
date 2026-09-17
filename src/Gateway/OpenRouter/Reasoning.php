<?php

namespace Laravel\Ai\Gateway\OpenRouter;

use Generator;
use Laravel\Ai\Gateway\ReasoningStream;
use Laravel\Ai\Streaming\Events\StreamEvent;

class Reasoning
{
    protected ReasoningStream $stream;

    protected string $text = '';

    /** @var array<array-key, array<string, mixed>> */
    protected array $details = [];

    protected int $unkeyed = 0;

    protected ?string $field = null;

    protected bool $sawDetails = false;

    public function __construct(string $invocationId)
    {
        $this->stream = new ReasoningStream($invocationId);
    }

    /**
     * @param  array<string, mixed>  $delta
     * @return Generator<int, StreamEvent>
     */
    public function process(array $delta): Generator
    {
        [$plain, $field] = static::plainFrom($delta);

        $this->field ??= $field;

        $details = $this->mergeDetails($delta);
        $reasoning = $plain !== '' ? $plain : $details;

        if ($reasoning !== '') {
            $this->text .= $reasoning;

            yield from $this->stream->push($reasoning);
        }

        if (static::startsAnswer($delta)) {
            yield from $this->close();
        }
    }

    /** @return Generator<int, StreamEvent> */
    public function close(): Generator
    {
        yield from $this->stream->close();
    }

    /** @return array<string, mixed> */
    public function providerContentBlocks(): array
    {
        return static::providerContentBlocksFor(
            $this->text,
            $this->sawDetails ? array_values($this->details) : null,
            $this->field,
        );
    }

    /**
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
     * @param  array<int, array<string, mixed>>|null  $details
     * @return array<string, mixed>
     */
    protected static function providerContentBlocksFor(string $reasoning, ?array $details, ?string $field): array
    {
        return array_filter([
            'reasoning_content' => $reasoning === '' ? null : $reasoning,
            'reasoning_details' => $details,
            'reasoning_field' => $reasoning === '' ? null : $field,
        ], fn (string|array|null $value): bool => $value !== null);
    }

    /** @param  array<array-key, mixed>  $providerContentBlocks */
    public static function replayableTextFrom(array $providerContentBlocks): ?string
    {
        $reasoning = $providerContentBlocks['reasoning_content'] ?? null;

        return is_string($reasoning) && $reasoning !== '' ? $reasoning : null;
    }

    /** @param  array<array-key, mixed>  $providerContentBlocks */
    public static function replayableFieldFrom(array $providerContentBlocks): string
    {
        $field = $providerContentBlocks['reasoning_field'] ?? null;

        return in_array($field, ['reasoning', 'reasoning_content'], true) ? $field : 'reasoning';
    }

    /**
     * @param  array<array-key, mixed>  $providerContentBlocks
     * @return array<int, array<string, mixed>>|null
     */
    public static function replayableDetailsFrom(array $providerContentBlocks): ?array
    {
        $details = $providerContentBlocks['reasoning_details'] ?? null;

        return is_array($details) ? array_values($details) : null;
    }

    /** @param  array<string, mixed>  $payload */
    protected static function textFrom(array $payload): string
    {
        $plain = static::plainFrom($payload)[0];

        return $plain !== '' ? $plain : static::detailsTextFrom($payload);
    }

    /**
     * @param  array<string, mixed>  $payload
     * @return array<int, array<string, mixed>>|null
     */
    protected static function detailsIn(array $payload): ?array
    {
        $details = $payload['reasoning_details'] ?? null;

        return is_array($details) ? array_values($details) : null;
    }

    /** @param  array<string, mixed>  $delta */
    protected function mergeDetails(array $delta): string
    {
        $details = static::detailsIn($delta);

        if ($details === null) {
            return '';
        }

        $this->sawDetails = true;
        $arrived = '';

        foreach ($details as $detail) {
            if (! is_array($detail)) {
                continue;
            }

            $arrived .= static::readableTextFrom($detail);
            $key = $this->detailKey($detail);

            if (! isset($this->details[$key])) {
                $this->details[$key] = $detail;

                continue;
            }

            $this->details[$key] = $this->mergeDetail($this->details[$key], $detail);
        }

        return $arrived;
    }

    /** @param  array<string, mixed>  $detail */
    protected function detailKey(array $detail): string
    {
        $identity = isset($detail['index'])
            ? 'index-'.$detail['index']
            : (isset($detail['id']) ? 'id-'.$detail['id'] : 'unkeyed-'.$this->unkeyed++);

        return ($detail['type'] ?? '').'#'.$identity;
    }

    /**
     * @param  array<string, mixed>  $captured
     * @param  array<string, mixed>  $fragment
     * @return array<string, mixed>
     */
    protected function mergeDetail(array $captured, array $fragment): array
    {
        $merged = [...$captured, ...$fragment];
        $field = match ($fragment['type'] ?? null) {
            'reasoning.text' => 'text',
            'reasoning.summary' => 'summary',
            'reasoning.encrypted' => 'data',
            default => null,
        };

        if ($field !== null && (isset($captured[$field]) || isset($fragment[$field]))) {
            $merged[$field] = (string) ($captured[$field] ?? '').(string) ($fragment[$field] ?? '');
        }

        return $merged;
    }

    /**
     * @param  array<string, mixed>  $payload
     * @return array{0: string, 1: string|null}
     */
    protected static function plainFrom(array $payload): array
    {
        foreach (['reasoning', 'reasoning_content'] as $field) {
            $reasoning = $payload[$field] ?? null;

            if (is_string($reasoning) && $reasoning !== '') {
                return [$reasoning, $field];
            }
        }

        return ['', null];
    }

    /** @param  array<string, mixed>  $payload */
    protected static function detailsTextFrom(array $payload): string
    {
        $text = '';

        foreach (static::detailsIn($payload) ?? [] as $detail) {
            if (is_array($detail)) {
                $text .= static::readableTextFrom($detail);
            }
        }

        return $text;
    }

    /** @param  array<string, mixed>  $detail */
    protected static function readableTextFrom(array $detail): string
    {
        $text = $detail['text'] ?? $detail['summary'] ?? null;

        return is_string($text) ? $text : '';
    }

    /** @param  array<string, mixed>  $delta */
    protected static function startsAnswer(array $delta): bool
    {
        $content = $delta['content'] ?? null;

        return (is_string($content) && $content !== '') || filled($delta['tool_calls'] ?? null);
    }
}
