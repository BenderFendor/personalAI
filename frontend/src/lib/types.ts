export interface Config {
  model: string
  temperature: number
  tools_enabled: boolean
  thinking_enabled: boolean
  show_thinking: boolean
  web_search_enabled: boolean
  auto_fetch_urls: boolean
  max_tool_iterations: number
  llm_provider: string
  gemini_model: string
}
