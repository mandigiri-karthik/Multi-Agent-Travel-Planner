from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

# res = tavily_search(query="Best hotels in Glasgow")
res = search_flights("Plan a 7 days glasgow trip from Banglore")
print(res)