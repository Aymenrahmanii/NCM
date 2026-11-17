# I tested a "31× faster" text generator on my laptop

Hazy Research recently released Next-Concept Modeling (NCM), a way of generating text that
doesn't go one token at a time. The headline number was up to 31× faster. I'm a final-year
data science student and the only GPU I have is the RTX 3050 in my gaming laptop (6 GB), so
my first question was simple: what does 31× mean for someone like me?

That question turned into a much bigger project than I expected. I benchmarked the official
models, built a small NCM from scratch, and eventually trained NCM and ordinary language
models side by side on the same data. All of it is in
[the repository](https://github.com/Aymenrahmanii/NCM), with every output and a script that
rebuilds each number.

## How NCM writes

A normal language model, the GPT kind (people call it autoregressive, or AR), writes left to
right. One token per step, so a 256-token story takes 256 steps.

NCM starts from the other end. It first predicts a single code that stands for the whole
text, then a handful of codes, then more, until there's one per token. All the codes at a
level come out at once, so the same story takes about 9 steps.

![The same story written one token per pass and one level per pass](assets/generation.gif)

Nine steps instead of 256 sounds like an easy win. It isn't quite, because each NCM step
works on the whole text, while an AR step only adds one token.

## First surprise: what the 31× is compared to

The 31× is measured against a diffusion language model, which needs 128 or more steps, on an
H100, with 1,024-token outputs. That's a reasonable comparison for the paper. It's just not
the one I had in my head, which was "faster than GPT".

Against same-size GPT-style models on my laptop, with caching and compilation on both sides,
it looked like this:

- One request at a time: NCM is 1.2–1.7× faster when both are compiled, and about even when
  neither is.
- 32 requests at once: NCM is 1.7–2× slower. AR models batch really well, and NCM's edge
  mostly comes from using GPU capacity that would otherwise sit idle.
- On the CPU: NCM is 1.1–1.7× slower. No idle GPU, no advantage.
- Long outputs (up to 1,024 tokens): 6–10× faster than GPT-2 small for a single request, but
  the gap doesn't widen as the text gets longer.

I also ran into a bug in the official caching code that made batched NCM look 2.5–4.5× slower
than it really is. Someone had already found it and opened a fix upstream; I ended up
confirming the same fix on full test sets. Honestly, measuring speed fairly took more of my
time than anything else.

## Second surprise: same training changes everything

The official checkpoints were trained by the NCM authors. The models I'd been comparing them
with were trained by other people, on other data. So when NCM lost on quality, I couldn't tell
whether the method was worse or the checkpoint simply had less training behind it.

The only way out was to train everything myself. I took all of TinyStories (2.1 million short
children's stories), one tokenizer, one optimizer and 200 million training positions per
model. On one side a 35M-parameter NCM, on the other five AR models from 1.9M to 35.9M. I
scored the stories with MAUVE, which measures how close generated text is to real text. Real
stories get 0.95.

![Quality against latency: NCM is far faster and far worse](../results/phase10_pareto.png)

| model | time per story | MAUVE |
|---|---|---|
| real stories | – | 0.95 |
| AR, 35.9M | 1,610 ms | 0.92 |
| AR, 1.9M | 694 ms | 0.59 |
| NCM, 35M | 61 ms | 0.006 |

NCM was about 20× faster and its stories fell apart. They'd open nicely ("Once upon a time,
there was a little girl named Lily.") and then stutter ("her her her mom her"). The smallest
AR model, 18 times smaller, wrote far better. I repeated it with three training seeds, seven
decoding settings and a second judge (a small LLM rating coherence), and nothing changed.

## Third surprise: my own design was part of the problem

The released 125M NCM writes decent TinyStories, with a MAUVE of 0.80. Mine got 0.006. A gap
that size made me suspect my own code before the method.

So I ran a test. I took real stories, turned them into NCM codes, kept the real codes for the
first few levels and let the model fill in the rest. With my original design it failed even
when it only had to write the very last level, with everything else handed to it. The reason
was my layout: I doubled the number of codes at each level, which left half the text to the
last one.

The official design spreads the text more evenly across levels. I rebuilt my tokenizer that
way and trained again. That fixed the last level: given the earlier levels, the model now
wrote it at close to human quality (0.94). The full stories, though, were still broken
(0.007). The problem had just moved up to the first levels, the ones that decide what the
story is about. The model turned out to be good at polishing a plan and bad at coming up with
one.

So my first explanation was partly wrong, and that one check taught me more than all the
extra runs I'd planned. The conclusion didn't change, the reason did: at this size it's the
method that loses to AR, not my version of it. Whether it catches up at 125M parameters and
beyond is something my laptop can't tell me.

## The useful part: fine-tuning

The most practical result was about adapting a released model to a new job. I fine-tuned the
official news-summary NCM to summarize chat conversations (the SAMSum dataset), and did the
same with BART, a standard summarizer, for comparison.

LoRA, which trains small add-on matrices instead of the whole model, got within about one
ROUGE-L point of full fine-tuning on average, at rank 16 on every layer and under 3% of the
weights. The setting you usually see in tutorials, rank 8 on two attention matrices, was 3–4
points short for both models. Worth checking before you reach for the default.

It didn't make NCM competitive, though. It stayed 14 points behind BART and made up a name
that wasn't in the conversation in about a third of its summaries.

## What I'm taking away

- When you see a speedup, find out what it's measured against. "31× faster" and "1.5× faster
  than what you'd actually use" can both be true.
- Speed only matters next to quality. Twenty times faster at writing broken text isn't twenty
  times better.
- If you want to compare two methods, train both yourself. Otherwise you're comparing
  checkpoints.
- When a result looks dramatic, go after your own implementation first. Mine was hiding part
  of the story.

All of this ran on one laptop GPU at small scale. Whether NCM can match AR at hundreds of
millions of parameters needs a bigger machine than mine. If you have one, the training and
evaluation code is ready to go.
