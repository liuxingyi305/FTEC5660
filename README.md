# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 
> to students: please fill your solution description here.

1. Chain design

The chain has two stages.

Stage 1 — Structured extraction (per receipt).**
Each receipt image is converted into a base64 data URL by the provided
`image_data_url()` helper and passed to a LangChain chain built as:

```
ChatPromptTemplate.from_messages(...) | ChatDeepSeek("deepseek-v4-flash-vision-exp", temperature=0)
```

The system prompt instructs the vision model to return **only** a JSON object
with four fields:

```json
{
  "subtotal": <number>,
  "discounts": [<positive numbers>],
  "rounding": <number>,
  "final_payment": <number>
}
```

Stage 2 — Local aggregation (across receipts).**
Python sums the per-receipt numbers using `Decimal` for exact currency math:

- `Query 1 = Σ final_payment`
- `Query 2 = Σ (subtotal + Σ |discounts|)`  (ROUNDING is not added back)

Because the vision model occasionally misses one discount line on receipts
with several stacked promotions, each receipt is queried up to 3 times and
the response whose `discounts` sum is the largest is kept. Missing a discount
is the dominant failure mode, and the model never fabricates extra discount
lines, so "best of 3 by max discount sum" reliably corrects the omission.

2. Chain diagram

```
receipt images (public_test/*.jpg)
        │
        ▼
image_data_url(path)  →  base64 data URL
        │
        ▼
ChatPromptTemplate
  system: "extract subtotal, discounts, rounding, final_payment as JSON"
  human:  [ text , image_url ]
        │
        ▼
ChatDeepSeek("deepseek-v4-flash-vision-exp", temperature=0)
        │
        ▼
JSON {subtotal, discounts, rounding, final_payment}
        │
        ▼
best-of-3 by max Σ|discounts|
        │
        ▼
Decimal aggregation across all receipts
        │
        ▼
{ QUERY_1: "HK$1974.30" , QUERY_2: "HK$2348.20" }
        │
        ▼
results.csv
```

3. Result

On the 7 public receipts, running `python3 hw1.py --image-folder public_test`
three times produced the same correct answers every time:

| Query | Response |
|-------|----------|
| How much money did I spend in total for these bills? | HK$1974.30 |
| How much would I have had to pay without the discount? | HK$2348.20 |

`results.csv` shows `correct` for both rows.


