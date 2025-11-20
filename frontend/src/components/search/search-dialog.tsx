"use client"

import { useState } from "react"
import { Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"

interface SearchDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SearchDialog({ open, onOpenChange }: SearchDialogProps) {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    try {
      const res = await fetch("http://localhost:8000/api/rag/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      })
      const data = await res.json()
      setResults(data)
    } catch (error) {
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Search Knowledge Base</DialogTitle>
          <DialogDescription>
            Search through indexed documents and web pages.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSearch} className="flex gap-2">
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search query..."
          />
          <Button type="submit" disabled={loading}>
            {loading ? "Searching..." : <Search className="h-4 w-4" />}
          </Button>
        </form>
        <ScrollArea className="h-[400px] mt-4">
          <div className="space-y-4">
            {results.map((result, i) => (
              <div key={i} className="p-4 border rounded-lg">
                <div className="font-medium mb-1">{result.metadata?.title || "Untitled"}</div>
                <div className="text-sm text-muted-foreground mb-2">{result.metadata?.source}</div>
                <div className="text-sm">{result.page_content}</div>
              </div>
            ))}
            {results.length === 0 && !loading && (
              <div className="text-center text-muted-foreground">No results found</div>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}
