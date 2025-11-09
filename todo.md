# Plan features

Don't save chat logs if nothing was typed.

Don't allow the user to backspace the You: part that shouldn't be deleteable in the ux for the terminal.

Allow the user to make a new chat by saving the current one and going to the next.

Fix the side bar to work as it doesn't right now and the crtl+] keybind doesn't work at all so commenlty rewrite that feature..

# Agentic tools

For the search and fetch_url make it so that it is really a two step process where it always searchs then uses fetch_url on all the urls that seems useful so make a two that uses both like search always use fetch_url but you can also use fetch_url by itself.

Also should more infomation about what the ai is taking so  with the fetch_url show all the snippets or chunks of the page that the AI is taking as well to the use in the terminal as part of it thought. have more transpart search results as well showing the full chain of thought of the model.

Also make sure the the aI is chunking and using chomadb for the vector search of the website add a tool to use chromadb and show that being used.

Look up best practices for using chromadb to chunk and vector index search results and full pages. Add citiations to the sources in all results as well when using fetch_url or search or new_search.