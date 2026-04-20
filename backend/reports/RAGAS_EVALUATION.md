# RAGAS Evaluation Summary

Generated: 2026-04-20T13:02:12.529678+00:00
Questions: 36

## Mean Metrics
- faithfulness: 0.8703703703703703
- answer_relevancy: 0.7043911849555344
- context_recall: 0.9074074074074073

## Category Checks
- ambiguous_rewrite: count=4, source_pass=None, safe_fail_pass=None, rewrite_pass=1.0, mean_iterations=2.5, decomp=3, web_search=0
- cross_document_confusion: count=3, source_pass=0.6666666666666666, safe_fail_pass=None, rewrite_pass=None, mean_iterations=3.0, decomp=3, web_search=0
- decomposition_multistep: count=3, source_pass=0.3333333333333333, safe_fail_pass=None, rewrite_pass=None, mean_iterations=3.0, decomp=3, web_search=0
- inference_grounding: count=3, source_pass=0.6666666666666666, safe_fail_pass=None, rewrite_pass=None, mean_iterations=2.3333333333333335, decomp=2, web_search=1
- precise_attribution: count=4, source_pass=0.5, safe_fail_pass=None, rewrite_pass=None, mean_iterations=1.5, decomp=1, web_search=0
- safe_fail_unanswerable: count=4, source_pass=None, safe_fail_pass=0.5, rewrite_pass=None, mean_iterations=1.25, decomp=0, web_search=0
- semantic_mismatch: count=3, source_pass=0.0, safe_fail_pass=None, rewrite_pass=None, mean_iterations=1.3333333333333333, decomp=1, web_search=0
- straightforward_factual: count=9, source_pass=0.7777777777777778, safe_fail_pass=None, rewrite_pass=None, mean_iterations=1.2222222222222223, decomp=1, web_search=0
- web_search_provenance: count=3, source_pass=None, safe_fail_pass=0.6666666666666666, rewrite_pass=None, mean_iterations=2.0, decomp=1, web_search=0
