"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { Config } from "@/lib/types"

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const [config, setConfig] = useState<Config | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (open) {
      fetchConfig()
    }
  }, [open])

  const fetchConfig = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/config")
      const data = await res.json()
      setConfig(data)
    } catch (error) {
      console.error("Failed to fetch config", error)
    }
  }

  const handleSave = async () => {
    if (!config) return
    setLoading(true)
    try {
      await fetch("http://localhost:8000/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      })
      onOpenChange(false)
    } catch (error) {
      console.error("Failed to save config", error)
    } finally {
      setLoading(false)
    }
  }

  if (!config) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Configure your AI assistant preferences.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model" className="text-right">
              Model
            </Label>
            <Input
              id="model"
              value={config.model}
              onChange={(e) => setConfig({ ...config, model: e.target.value })}
              className="col-span-3"
            />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="temp" className="text-right">
              Temperature
            </Label>
            <div className="col-span-3 flex items-center gap-2">
              <Slider
                id="temp"
                min={0}
                max={1}
                step={0.1}
                value={[config.temperature]}
                onValueChange={(vals) => setConfig({ ...config, temperature: vals[0] })}
              />
              <span className="w-8 text-sm">{config.temperature}</span>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="tools">Enable Tools</Label>
            <Switch
              id="tools"
              checked={config.tools_enabled}
              onCheckedChange={(checked) => setConfig({ ...config, tools_enabled: checked })}
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="thinking">Enable Thinking</Label>
            <Switch
              id="thinking"
              checked={config.thinking_enabled}
              onCheckedChange={(checked) => setConfig({ ...config, thinking_enabled: checked })}
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="show_thinking">Show Thinking</Label>
            <Switch
              id="show_thinking"
              checked={config.show_thinking}
              onCheckedChange={(checked) => setConfig({ ...config, show_thinking: checked })}
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="web_search">Web Search</Label>
            <Switch
              id="web_search"
              checked={config.web_search_enabled}
              onCheckedChange={(checked) => setConfig({ ...config, web_search_enabled: checked })}
            />
          </div>
        </div>
        <DialogFooter>
          <Button onClick={handleSave} disabled={loading}>
            {loading ? "Saving..." : "Save changes"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
