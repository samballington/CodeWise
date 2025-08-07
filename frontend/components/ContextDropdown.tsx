'use client'

import React, { useState } from 'react'
import { ChevronDown, ChevronUp, Search, FileText, Zap, Loader2 } from 'lucide-react'
import { useThemeStore, useContextStore } from '../lib/store'

interface ToolCall {
  tool: string
  input: string
  output: string
  status: 'running' | 'completed' | 'error'
}

interface ContextDropdownProps {
  toolCalls?: ToolCall[]
  messageId: string
  contextData?: {
    sources?: string[]
    chunksFound?: number
    filesAnalyzed?: number
    query?: string
  }
}

export function ContextDropdown({ toolCalls = [], messageId, contextData }: ContextDropdownProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const { isDarkMode } = useThemeStore()

  // Debug logging
  console.log('ContextDropdown render:', {
    messageId,
    toolCalls: toolCalls.length,
    contextData,
    hasContextData: !!(contextData && contextData.sources && contextData.sources.length > 0)
  })

  const hasContext = toolCalls.length > 0 || (contextData && contextData.sources && contextData.sources.length > 0)
  
  if (!hasContext) {
    return null
  }

  const getToolIcon = (toolName: string) => {
    switch (toolName.toLowerCase()) {
      case 'smart_search':
        return <Search className="w-3 h-3" />
      case 'examine_files':
        return <FileText className="w-3 h-3" />
      case 'analyze_relationships':
        return <Zap className="w-3 h-3" />
      default:
        return <FileText className="w-3 h-3" />
    }
  }

  return (
    <div className={`border rounded-lg mt-2 transition-colors ${
      isDarkMode 
        ? 'border-slate-600 bg-slate-800' 
        : 'border-gray-200 bg-gray-50'
    }`}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`w-full px-3 py-2 text-left flex items-center justify-between text-sm transition-colors ${
          isDarkMode
            ? 'text-gray-300 hover:text-gray-100 hover:bg-slate-700'
            : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
        }`}
      >
        <div className="flex items-center gap-2">
          <span className="font-medium">Context & Tool Calls</span>
          {toolCalls.length > 0 && (
            <span className={`text-xs px-2 py-1 rounded-full ${
              isDarkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-700'
            }`}>
              {toolCalls.length}
            </span>
          )}
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4" />
        ) : (
          <ChevronDown className="w-4 h-4" />
        )}
      </button>

      {isExpanded && (
        <div className={`border-t px-3 py-3 space-y-3 ${
          isDarkMode ? 'border-slate-600' : 'border-gray-200'
        }`}>
          {/* Context Information */}
          {contextData && contextData.sources && contextData.sources.length > 0 && (
            <div className={`p-3 rounded-lg ${
              isDarkMode ? 'bg-slate-700' : 'bg-white'
            }`}>
              <h4 className={`text-sm font-medium mb-2 ${
                isDarkMode ? 'text-gray-200' : 'text-gray-800'
              }`}>
                Query Context
              </h4>
              {contextData.query && (
                <p className={`text-xs mb-2 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  <strong>Query:</strong> {contextData.query}
                </p>
              )}
              {contextData.chunksFound !== undefined && (
                <p className={`text-xs mb-2 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  <strong>Chunks Found:</strong> {contextData.chunksFound}
                </p>
              )}
              {contextData.filesAnalyzed !== undefined && (
                <p className={`text-xs mb-2 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  <strong>Files Analyzed:</strong> {contextData.filesAnalyzed}
                </p>
              )}
              <div>
                <p className={`text-xs font-medium mb-1 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  Sources Analyzed:
                </p>
                <div className="flex flex-wrap gap-1">
                  {contextData.sources.slice(0, 5).map((source, sourceIdx) => (
                    <span
                      key={sourceIdx}
                      className={`text-xs px-2 py-1 rounded ${
                        isDarkMode 
                          ? 'bg-slate-600 text-gray-300' 
                          : 'bg-gray-200 text-gray-700'
                      }`}
                    >
                      {source}
                    </span>
                  ))}
                  {contextData.sources.length > 5 && (
                    <span className={`text-xs px-2 py-1 rounded ${
                      isDarkMode 
                        ? 'bg-slate-600 text-gray-400' 
                        : 'bg-gray-200 text-gray-600'
                    }`}>
                      +{contextData.sources.length - 5} more
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Tool Calls */}
          {toolCalls.length > 0 && (
            <div className="space-y-2">
              <h4 className={`text-sm font-medium ${
                isDarkMode ? 'text-gray-200' : 'text-gray-800'
              }`}>
                Tool Executions
              </h4>
              {toolCalls.map((toolCall, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-lg border ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600' 
                      : 'bg-white border-gray-200'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    {getToolIcon(toolCall.tool)}
                    <span className={`text-sm font-medium ${
                      isDarkMode ? 'text-gray-200' : 'text-gray-800'
                    }`}>
                      {toolCall.tool}
                    </span>
                    <span className={`text-xs px-2 py-1 rounded-full flex items-center gap-1 ${
                      toolCall.status === 'completed'
                        ? isDarkMode ? 'bg-green-900 text-green-300' : 'bg-green-100 text-green-700'
                        : toolCall.status === 'error'
                        ? isDarkMode ? 'bg-red-900 text-red-300' : 'bg-red-100 text-red-700'
                        : isDarkMode ? 'bg-yellow-900 text-yellow-300' : 'bg-yellow-100 text-yellow-700'
                    }`}>
                      {toolCall.status === 'running' && <Loader2 className="w-3 h-3 animate-spin" />}
                      {toolCall.status}
                    </span>
                  </div>
                  <div className={`text-xs mb-2 ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-600'
                  }`}>
                    <strong>Input:</strong> {toolCall.input.substring(0, 100)}
                    {toolCall.input.length > 100 && '...'}
                  </div>
                  {toolCall.output && toolCall.status !== 'running' && (
                    <div className={`text-xs ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-600'
                    }`}>
                      <strong>Output:</strong> {toolCall.output.substring(0, 200)}
                      {toolCall.output.length > 200 && '...'}
                    </div>
                  )}
                  {toolCall.status === 'running' && (
                    <div className={`text-xs flex items-center gap-2 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-600'
                    }`}>
                      <Loader2 className="w-3 h-3 animate-spin" />
                      <span>Executing...</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}