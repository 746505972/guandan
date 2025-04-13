import pstats

stats = pstats.Stats("profile_results.prof")
stats.strip_dirs().sort_stats("cumulative").print_stats(20)  # 显示前 20 个最慢的函数
