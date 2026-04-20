# RAGAS Evaluation Summary

Generated: 2026-04-20T12:10:47.231627+00:00
Questions: 36

## Mean Metrics
- faithfulness: 0.6071428571428571
- answer_relevancy: 0.6723383029190753
- context_recall: 0.7878787878787878

## Category Checks
- ambiguous_rewrite: count=4, source_pass=None, safe_fail_pass=None, rewrite_pass=1.0, mean_iterations=1.5, decomp=1, web_search=0
- cross_document_confusion: count=3, source_pass=1.0, safe_fail_pass=None, rewrite_pass=None, mean_iterations=1.6666666666666667, decomp=1, web_search=0
- decomposition_multistep: count=3, source_pass=0.6666666666666666, safe_fail_pass=None, rewrite_pass=None, mean_iterations=2.3333333333333335, decomp=2, web_search=0
- inference_grounding: count=3, source_pass=0.6666666666666666, safe_fail_pass=None, rewrite_pass=None, mean_iterations=1.6666666666666667, decomp=1, web_search=0
- precise_attribution: count=4, source_pass=0.75, safe_fail_pass=None, rewrite_pass=None, mean_iterations=1.0, decomp=0, web_search=0
- safe_fail_unanswerable: count=4, source_pass=None, safe_fail_pass=0.75, rewrite_pass=None, mean_iterations=2.0, decomp=2, web_search=2
- semantic_mismatch: count=3, source_pass=0.0, safe_fail_pass=None, rewrite_pass=None, mean_iterations=1.0, decomp=0, web_search=0
- straightforward_factual: count=9, source_pass=0.6666666666666666, safe_fail_pass=None, rewrite_pass=None, mean_iterations=1.0, decomp=0, web_search=1
- web_search_provenance: count=3, source_pass=None, safe_fail_pass=0.0, rewrite_pass=None, mean_iterations=4.0, decomp=1, web_search=3
