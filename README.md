#  Utilizing SAE Features for Activation Steering

Below is a rough flowchart of different things that need to be done, mostly prerequisites for the actual research.

Critical steps that need to work for the project to make sense are highlighted in red, with comments below the flowchart.

```mermaid
flowchart TD

%% Node connections
subgraph For the mid-may Presentation
    verify_saes --> dataset
    dataset --> compute_svs
    dataset --> eval_gpt2
    compute_svs --> eval_svs
    eval_gpt2 --> eval_svs

    verify_saes --> run_saes
    dataset --> eval_saes
    run_saes --> eval_saes
end

eval_svs --> q1
eval_saes --> q1
eval_saes --> q2

%% Node titles
verify_saes[Verify that GPT-2 small SAEs are publicly available]
dataset[Prepare ca. 3 contrastive datasets\nusing Rimsky et al.]
compute_svs[Compute SVs\nusing Rimsky et al.]
eval_gpt2[Eval base GPT-2 on the dataset]
run_saes[Actually download the SAEs\nand make them work]
eval_svs[Eval SVs, compare with base GPT-2\nfor plot & eval inspo see Rimsky et al.]
eval_saes[
    Eval SAEs on the dataset.
    Check that there is a meaningful difference
    in which features fire in the POS v. NEG tasks.
]

%% Questions
q1[
    Q1: Are SVs aligned with features?
]
q2[
    Q2: Do we see different groups
    of contrastive feature pairs?
]

%% Styling for the critical steps
classDef critical fill:#FEE2E2,stroke:#333,color:#EF4444,stroke:#EF4444

%% List of critical steps
class verify_saes,eval_svs,eval_saes critical
```

## Comments on critical steps

If we fail to reproduce Rimsky — i.e. if our steering vectors don't steer the model — we will have to rethink the whole plan. Maybe GPT-2 is hard to steer on artificial datasets or something.

The SAEs should perform very similarly to base GPT-2 on our datasets. If they don't, there's something wrong with them (or their interaction with the dataset).

If all features fire similarly on the positive and negative examples, we're doing something wrong — or maybe the features are too low-level to capture any high-level ideas like "refusal".

## Research questions

**Q1:  Are SVs aligned with features?** I'd expect a steering vector to be an average of multiple related features that all capture the idea of the steering vector but in different contexts / with slightly different flavours.

**Q2: Do we see different groups of contrastive feature pairs?** This is related to Q1. Say we run SAEs on the contrastive dataset we prepared — can we further "split" the dataset to smaller semantic chunks in which different features fire? For example, in a refusal dataset, can we identify something like "refusal because dangerous" and "refusal because lack of understanding" groups of tasks in which completely different features do the "refusal" part of the job?

Other research questions worth considering:
- Generally inspect how features behave on the contrastive dataset. Are there any "opposite" features?
- Can we use the features for steering?
- How do steering features compare to SVs in performance?
- Can we compute the steering features without the contrastive dataset? E.g. maybe there's a patter
