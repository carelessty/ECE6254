# Analysis of Research Papers on Privacy Risk Detection

## Paper 1: "Measuring, Modeling, and Helping People Account for Privacy Risks in Online Self-Disclosures with AI"

This paper focuses on helping users identify and mitigate privacy risks in online self-disclosures, particularly on pseudonymous platforms like Reddit. The authors conducted a study with 21 Reddit users to understand how NLP disclosure detection models can help users make more informed decisions about their privacy.

Key findings:
- Users on pseudonymous platforms like Reddit often disclose highly sensitive information
- While self-disclosure has benefits (emotional support, community building), it also carries privacy risks
- Previous NLP tools for privacy risk detection have not been evaluated with the actual users they aim to protect
- The authors developed a span-level self-disclosure detection model that highlights potentially risky content
- User study showed that the model helped users identify risks they were unaware of and encouraged self-reflection
- For AI to be useful in privacy decision-making, it must account for posting context, disclosure norms, and users' lived threat models

The paper emphasizes that NLP-based disclosure detection tools should be designed to help users make informed decisions about online self-disclosures, rather than simply telling users to avoid self-disclosure altogether.

## Paper 2: "Reducing Privacy Risks in Online Self-Disclosures with Language Models"

This paper introduces a comprehensive approach to privacy risk detection and mitigation in user-generated content through two main components:

1. **Self-disclosure detection**: Identifying potentially risky self-disclosures in text
2. **Self-disclosure abstraction**: Rephrasing disclosures into less specific terms while preserving utility

Key methodology details:
- Development of a taxonomy with 19 self-disclosure categories (13 demographic attributes and 6 personal experiences)
- Creation of a high-quality dataset with human annotations on 2.4K Reddit posts, covering 4.8K varied self-disclosures
- Fine-tuning a language model to identify self-disclosures in text, achieving over 65% partial span F1
- Conducting an HCI user study with 21 Reddit users to validate real-world applicability (82% had positive outlook)
- Introduction of a novel task of self-disclosure abstraction to rephrase disclosures in ways that reduce privacy risks while maintaining utility
- The best model, distilled on GPT-3.5, generated abstractions that increased privacy moderately (scoring 3.2/5) while preserving high utility (scoring 4/5)
- The model achieved 80% accuracy, comparable to GPT-3.5

The paper also introduces a task of rating the importance of self-disclosure in understanding context, which helps users determine which disclosures to abstract.

## Common Themes and Methodological Insights

Both papers highlight:
1. The tension between the benefits of self-disclosure and associated privacy risks
2. The need for user-centered approaches that help users make informed decisions rather than paternalistic tools
3. The importance of evaluating models with actual end-users

The methodology for privacy risk detection involves:
1. Creating a taxonomy of self-disclosure categories
2. Building an annotated dataset of self-disclosures from platforms like Reddit
3. Fine-tuning language models to detect potentially risky content at the span level
4. Evaluating models both technically (F1 scores) and through user studies
5. Extending detection with abstraction capabilities to help users reduce privacy risks while maintaining message utility

These insights will be crucial for adapting the DeepSeek-R1-Distill-Qwen-1.5B model for privacy risk detection in user-generated content.
