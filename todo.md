# Plan features

Don't save chat logs if nothing was typed.

Don't allow the user to backspace the You: part that shouldn't be deleteable in the ux for the terminal.

Allow the user to make a new chat by saving the current one and going to the next.

Fix the side bar to work as it doesn't right now and the crtl+] keybind doesn't work at all so commenlty rewrite that feature..

Also research how to get the ollama context size to truly work as the one we have right now doesn't do anything.

I need to get the context like current context window size from the model itself then when responses are send and reserviced I need that context window to update so like how much of the current context window is used.