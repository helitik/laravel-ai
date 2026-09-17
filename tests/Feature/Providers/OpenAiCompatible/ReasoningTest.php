<?php

use Illuminate\Support\Facades\Http;
use Laravel\Ai\Streaming\Events\ReasoningDelta;
use Laravel\Ai\Streaming\Events\ReasoningEnd;
use Laravel\Ai\Streaming\Events\ReasoningStart;
use Tests\Fixtures\Tools\FixedNumberGenerator;

use function Laravel\Ai\agent;

beforeEach(function (): void {
    configureOpenAiCompatible();
});

function fakeOpenAiCompatibleToolCallResponse(array $message = [])
{
    return Http::response([
        'id' => 'chatcmpl-tool-123',
        'object' => 'chat.completion',
        'model' => 'local-model',
        'choices' => [[
            'index' => 0,
            'message' => [
                'role' => 'assistant',
                'content' => null,
                'tool_calls' => [[
                    'id' => 'call_1',
                    'type' => 'function',
                    'function' => ['name' => 'FixedNumberGenerator', 'arguments' => '{}'],
                ]],
                ...$message,
            ],
            'finish_reason' => 'tool_calls',
        ]],
        'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
    ]);
}

function openAiCompatibleToolCallMessage(): ?array
{
    $messages = json_decode(Http::recorded()[1][0]->body(), true)['messages'];

    return collect($messages)->first(fn (array $message): bool => isset($message['tool_calls']));
}

test('emits reasoning events while streaming', function (): void {
    Http::fake(['*' => Http::response(
        body: implode("\n\n", [
            'data: '.json_encode(['choices' => [['index' => 0, 'delta' => ['role' => 'assistant', 'reasoning' => 'Hmm, let me think.']]]]),
            'data: '.json_encode(['choices' => [['index' => 0, 'delta' => ['content' => 'Hello']]]]),
            'data: '.json_encode(['choices' => [['index' => 0, 'delta' => [], 'finish_reason' => 'stop']]]),
            'data: [DONE]',
        ])."\n\n",
        headers: ['Content-Type' => 'text/event-stream'],
    )]);

    $events = [];

    foreach (agent()->stream('Hello', provider: 'openai-compatible') as $event) {
        $events[] = $event;
    }

    $deltas = array_values(array_filter($events, fn ($e): bool => $e instanceof ReasoningDelta));

    expect($deltas)->toHaveCount(1)
        ->and($deltas[0]->delta)->toBe('Hmm, let me think.');

    $types = array_map(fn ($e): string => $e::class, $events);

    expect($types)->toContain(ReasoningStart::class)->toContain(ReasoningEnd::class);
});

test('replays reasoning under the reasoning field current vllm answers on', function (): void {
    Http::fake(['*' => Http::sequence([
        fakeOpenAiCompatibleToolCallResponse(['reasoning' => 'I need the generator for this.']),
        fakeOpenAiCompatibleResponse('The number is 72019'),
    ])]);

    agent(tools: [new FixedNumberGenerator])->prompt('Give me a number', provider: 'openai-compatible');

    expect(openAiCompatibleToolCallMessage())
        ->reasoning->toBe('I need the generator for this.')
        ->not->toHaveKey('reasoning_content');
});

test('replays reasoning under the legacy reasoning_content field when the upstream uses it', function (): void {
    Http::fake(['*' => Http::sequence([
        fakeOpenAiCompatibleToolCallResponse(['reasoning_content' => 'I need the generator for this.']),
        fakeOpenAiCompatibleResponse('The number is 72019'),
    ])]);

    agent(tools: [new FixedNumberGenerator])->prompt('Give me a number', provider: 'openai-compatible');

    expect(openAiCompatibleToolCallMessage())
        ->reasoning_content->toBe('I need the generator for this.')
        ->not->toHaveKey('reasoning');
});

test('sends no reasoning fields when the upstream returned none', function (): void {
    Http::fake(['*' => Http::sequence([
        fakeOpenAiCompatibleToolCallResponse(),
        fakeOpenAiCompatibleResponse('The number is 72019'),
    ])]);

    agent(tools: [new FixedNumberGenerator])->prompt('Give me a number', provider: 'openai-compatible');

    expect(openAiCompatibleToolCallMessage())
        ->not->toHaveKey('reasoning')
        ->not->toHaveKey('reasoning_content');
});
