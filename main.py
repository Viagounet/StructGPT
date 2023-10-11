from structures.news import ConflictAnalysis


content = "my-content"

analysis = ConflictAnalysis(content).source("Fill the relevant information. However, chance all the names/event by transposing this information as if it happened on mars.")
analysis = analysis.act("situation_summary", "translate to french").act("situation_summary", "write in a bullet point format")
print(analysis.situation_summary)