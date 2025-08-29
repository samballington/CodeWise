'use client'

import React, { useState, useRef, useEffect } from 'react'
import { ChevronDown, Brain, Zap, Lightbulb } from 'lucide-react'
import { useThemeStore } from '../lib/store'

interface ModelSelectorProps {
  selectedModel: string
  onModelChange: (model: string) => void
  disabled?: boolean
}

interface ModelOption {
  id: string
  name: string
  description: string
  icon: React.ReactNode
  capabilities: string[]
}

const MODEL_OPTIONS: ModelOption[] = [
  {
    id: 'gpt-oss-120b',
    name: 'GPT-OSS-120B',
    description: 'Balanced performance for general coding tasks',
    icon: <Brain className="w-4 h-4" />,
    capabilities: ['General Coding', 'Problem Solving', 'Fast Response']
  },
  {
    id: 'qwen-3-coder-480b',
    name: 'Qwen-3-Coder-480B',
    description: 'Specialized for complex coding and software architecture',
    icon: <Zap className="w-4 h-4" />,
    capabilities: ['Advanced Coding', 'Architecture Design', 'Code Analysis']
  },
  {
    id: 'qwen-3-235b-a22b-thinking-2507',
    name: 'Qwen-3-Thinking-235B',
    description: 'Deep reasoning model for complex problem solving',
    icon: <Lightbulb className="w-4 h-4" />,
    capabilities: ['Deep Reasoning', 'Complex Analysis', 'Step-by-step Thinking']
  }
]

export default function ModelSelector({ selectedModel, onModelChange, disabled = false }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const { isDarkMode } = useThemeStore()

  const selectedOption = MODEL_OPTIONS.find(option => option.id === selectedModel) || MODEL_OPTIONS[0]

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleSelect = (modelId: string) => {
    onModelChange(modelId)
    setIsOpen(false)
  }

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-medium transition-all duration-200 ${
          disabled
            ? 'opacity-50 cursor-not-allowed'
            : 'hover:border-primary/50'
        } ${
          isDarkMode
            ? 'bg-background-secondary border-border text-text-primary hover:bg-background-secondary/80'
            : 'bg-white border-gray-200 text-gray-900 hover:bg-gray-50'
        }`}
      >
        <div className="flex items-center gap-2">
          {selectedOption.icon}
          <span className="hidden sm:inline">{selectedOption.name}</span>
          <span className="sm:hidden">Model</span>
        </div>
        <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className={`absolute top-full mt-2 left-0 right-0 min-w-80 rounded-lg shadow-lg border z-50 ${
          isDarkMode
            ? 'bg-background-secondary border-border'
            : 'bg-white border-gray-200'
        }`}>
          <div className="py-2">
            {MODEL_OPTIONS.map((option) => (
              <button
                key={option.id}
                onClick={() => handleSelect(option.id)}
                className={`w-full px-4 py-3 text-left transition-colors duration-200 ${
                  option.id === selectedModel
                    ? isDarkMode
                      ? 'bg-primary/20 text-primary'
                      : 'bg-primary/10 text-primary'
                    : isDarkMode
                    ? 'hover:bg-background-tertiary text-text-primary'
                    : 'hover:bg-gray-50 text-gray-900'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className={`mt-0.5 ${
                    option.id === selectedModel 
                      ? 'text-primary' 
                      : isDarkMode 
                        ? 'text-text-secondary' 
                        : 'text-gray-500'
                  }`}>
                    {option.icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{option.name}</span>
                      {option.id === selectedModel && (
                        <div className={`w-2 h-2 rounded-full ${
                          isDarkMode ? 'bg-primary' : 'bg-primary'
                        }`} />
                      )}
                    </div>
                    <p className={`text-xs mt-1 ${
                      isDarkMode ? 'text-text-secondary' : 'text-gray-600'
                    }`}>
                      {option.description}
                    </p>
                    <div className="flex flex-wrap gap-1 mt-2">
                      {option.capabilities.map((capability, index) => (
                        <span
                          key={index}
                          className={`px-2 py-0.5 text-xs rounded-full ${
                            isDarkMode
                              ? 'bg-background-tertiary text-text-secondary'
                              : 'bg-gray-100 text-gray-600'
                          }`}
                        >
                          {capability}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}