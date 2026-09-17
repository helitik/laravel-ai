<?php

namespace Laravel\Ai\Gateway;

use Generator;
use Illuminate\Support\Str;
use Laravel\Ai\Streaming\Events\ReasoningDelta;
use Laravel\Ai\Streaming\Events\ReasoningEnd;
use Laravel\Ai\Streaming\Events\ReasoningStart;
use Laravel\Ai\Streaming\Events\StreamEvent;

class ReasoningStream
{
    protected string $reasoningId = '';

    protected bool $open = false;

    public function __construct(protected string $invocationId) {}

    /** @return Generator<int, StreamEvent> */
    public function push(string $delta): Generator
    {
        if ($delta === '') {
            return;
        }

        if (! $this->open) {
            $this->open = true;
            $this->reasoningId = $this->generateEventId();

            yield $this->event(new ReasoningStart($this->generateEventId(), $this->reasoningId, time()));
        }

        yield $this->event(new ReasoningDelta($this->generateEventId(), $this->reasoningId, $delta, time()));
    }

    /** @return Generator<int, StreamEvent> */
    public function close(): Generator
    {
        if (! $this->open) {
            return;
        }

        $this->open = false;

        yield $this->event(new ReasoningEnd($this->generateEventId(), $this->reasoningId, time()));

        $this->reasoningId = '';
    }

    protected function event(StreamEvent $event): StreamEvent
    {
        return $event->withInvocationId($this->invocationId);
    }

    protected function generateEventId(): string
    {
        return strtolower((string) Str::uuid7());
    }
}
