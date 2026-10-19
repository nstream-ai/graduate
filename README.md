# Graduate

**Prove when your small model is ready to take over.**

> **Code coming soon.** The Graduate library is in a private research preview. This repo holds the project's website for now; the code will be published here or linked from here. **Watch this repo** (top right → Watch → Custom → Releases) to get a notification when it lands.

**Website:** https://nstream-ai.github.io/graduate-site/

## What it is

An AI agent asks its model many small questions per conversation: which tool to use, what to fill in, what to reply. Many of them are routine, and a small, cheap model can answer them for a fraction of the cost. Graduate does two things:

1. **Distil:** train a small model on your agent's own logs and hand back a LoRA adapter.
2. **Graduate:** give a report card, per slice of traffic, saying which slices the small model can take over, with a statistical promise that holds up in real agent traffic: linked calls within a conversation, people checking results every day, several right answers, and imperfect labels.

## Where it stands

The statistics are built and checked on simulated data and on a real week of agent traffic. Trained students are improving (from 23% to about 80% exact tool-call agreement with the big model), but no student has fully graduated yet. The website explains what that means and why.

## This repo

`index.html` is the whole site: one static page, no build step, published with GitHub Pages from the `main` branch.
