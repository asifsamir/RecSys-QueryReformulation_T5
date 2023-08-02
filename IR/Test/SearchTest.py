from IR.Searcher.Index_Searcher import Index_Searcher

searcher = Index_Searcher()

results = searcher.search("xmpp null pointer exception", top_K_results=10)

print(results)
