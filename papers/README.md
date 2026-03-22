# Downloaded Papers

## Core Papers (Most Directly Relevant)

1. **Cramming 1568 Tokens into a Single Vector and Back Again** (2502.13063)
   - Authors: Kuratov, Arkhipov, Bulatov, Burtsev (2025, ACL Oral)
   - File: `2502.13063_cramming_tokens_single_vector.pdf`
   - Why relevant: Directly measures hidden state capacity across models and encoding schemes

2. **Toy Models of Superposition** (2209.10652)
   - Authors: Elhage, Hume, Olsson et al. (Anthropic, 2022)
   - File: `2209.10652_toy_models_superposition.pdf`
   - Why relevant: Explains how models store more features than dimensions via superposition

3. **Eliciting Latent Predictions with the Tuned Lens** (2303.08112)
   - Authors: Belrose, Ostrovsky, McKinney et al. (2023)
   - File: `2303.08112_tuned_lens.pdf`
   - Why relevant: Key method for decoding hidden states via learned affine + unembedding

4. **Future Lens: Anticipating Subsequent Tokens from a Single Hidden State** (2311.04897)
   - Authors: Pal, Sun, Yuan, Wallace, Bau (CoNLL 2023)
   - File: `2311.04897_future_lens.pdf`
   - Why relevant: Shows hidden states encode multi-token future information

## Capacity and Memorization

5. **How Much Do Language Models Memorize?** (2505.24832)
   - Authors: Morris et al. (2025)
   - File: `2505.24832_how_much_llms_memorize.pdf`
   - Why relevant: ~3.5-4 bits per parameter capacity measurement

6. **On the Optimal Memorization Capacity of Transformers** (2409.17677)
   - Authors: Kajitsuka, Sato (2024)
   - File: `2409.17677_optimal_memorization_capacity_transformers.pdf`
   - Why relevant: Theoretical capacity bounds

7. **Upper and Lower Memory Capacity Bounds of Transformers** (2405.13718)
   - Authors: (2024)
   - File: `2405.13718_upper_lower_memory_capacity_transformers.pdf`
   - Why relevant: Formal capacity bounds for next-token prediction

## Bottlenecks and Representation

8. **Breaking the Softmax Bottleneck** (1711.03953)
   - Authors: Yang, Dai, Salakhutdinov, Cohen (ICLR 2018)
   - File: `1711.03953_softmax_bottleneck.pdf`
   - Why relevant: Hidden state dimension limits expressiveness through rank-deficient softmax

9. **nGPT: Normalized Transformer on the Hypersphere** (2410.01131)
   - Authors: Loshchilov et al. (2024)
   - File: `2410.01131_ngpt_hypersphere.pdf`
   - Why relevant: Vector walks on hypersphere, directional encoding

10. **Linear Representation Hypothesis** (2311.03658)
    - Authors: Park et al. (2023)
    - File: `2311.03658_linear_representation_hypothesis.pdf`
    - Why relevant: Concepts as linear directions in representation space

11. **Representation Engineering** (2310.01405)
    - Authors: Zou et al. (2023)
    - File: `2310.01405_representation_engineering.pdf`
    - Why relevant: Reading/controlling hidden state directions for concepts

## Interpretability and Probing

12. **Analyzing Transformers in Embedding Space** (2209.02535)
    - Authors: Dar, Geva, Gupta, Berant (ACL 2023)
    - File: `2209.02535_analyzing_transformers_embedding_space.pdf`
    - Why relevant: Projecting all parameters into vocabulary space

13. **Patchscopes** (2401.06102)
    - Authors: Ghandeharioun et al. (Google, 2024)
    - File: `2401.06102_patchscopes.pdf`
    - Why relevant: Unified framework for inspecting hidden representations

14. **BERT Rediscovers the Classical NLP Pipeline** (1905.05950)
    - Authors: Tenney, Das, Pavlick (ACL 2019)
    - File: `1905.05950_bert_classical_nlp_pipeline.pdf`
    - Why relevant: What linguistic information is encoded at each layer

15. **Information-Theoretic Probing with MDL** (2003.12298)
    - Authors: Voita, Titov (EMNLP 2020)
    - File: `2003.12298_probing_mdl.pdf`
    - Why relevant: Information-theoretic framework for measuring hidden state content

## Knowledge Storage

16. **Transformer Feed-Forward Layers Are Key-Value Memories** (2012.14913)
    - Authors: Geva, Schuster, Berant, Levy (EMNLP 2021)
    - File: `2012.14913_ffn_key_value_memories.pdf`
    - Why relevant: How information is stored in FFN layers

17. **Knowledge Neurons in Pretrained Transformers** (2104.08696)
    - Authors: Dai et al. (ACL 2022)
    - File: `2104.08696_knowledge_neurons.pdf`
    - Why relevant: Specific neurons expressing factual knowledge

## Steganography and Encoding

18. **A Watermark for Large Language Models** (2301.10226)
    - Authors: Kirchenbauer et al. (2023)
    - File: `2301.10226_watermark_llm.pdf`
    - Why relevant: Vocabulary-based encoding scheme for hiding bits in token selection

19. **The Steganographic Potentials of Language Models** (2505.03439)
    - Authors: (2025)
    - File: `2505.03439_steganographic_potentials_lm.pdf`
    - Why relevant: LLMs developing covert encoding schemes

## Communication and Circuits

20. **In-context Learning and Induction Heads** (2209.11895)
    - Authors: Olsson, Elhage et al. (Anthropic, 2022)
    - File: `2209.11895_incontext_learning_induction_heads.pdf`
    - Why relevant: Residual stream as communication channel between components
