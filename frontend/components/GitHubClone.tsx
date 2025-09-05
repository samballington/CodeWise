'use client'

import React, { useState } from 'react'
import { Github, Globe } from 'lucide-react'
import { useThemeStore } from '../lib/store'

interface GitHubCloneProps {
  onCloneSuccess?: () => void
}

export default function GitHubClone({ onCloneSuccess }: GitHubCloneProps) {
  const [repoUrl, setRepoUrl] = useState('')
  const [isCloning, setIsCloning] = useState(false)
  const { isDarkMode } = useThemeStore()

  const cloneRepo = async () => {
    if (!repoUrl.trim()) return
    
    setIsCloning(true)
    try {
      const response = await fetch('http://localhost:8000/projects/clone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: repoUrl.trim() })
      })
      
      const result = await response.json()
      
      if (response.ok) {
        alert(`Repository cloned successfully as '${result.project_name}'!`)
        setRepoUrl('')
        onCloneSuccess?.()
        // Refresh projects
        window.location.reload()
      } else {
        alert(`Failed to clone repository: ${result.detail}`)
      }
    } catch (error) {
      alert(`Error cloning repository: ${error}`)
    } finally {
      setIsCloning(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isCloning && repoUrl.trim()) {
      cloneRepo()
    }
  }

  return (
    <div className={`p-6 rounded-lg border ${isDarkMode ? 'bg-background-secondary border-border' : 'bg-white border-gray-200'}`}>
      <div className="text-center mb-6">
        <Github className="w-12 h-12 mx-auto mb-4 text-primary" />
        <h3 className={`text-lg font-semibold mb-2 ${isDarkMode ? 'text-text-primary' : 'text-gray-900'}`}>
          Clone GitHub Repository
        </h3>
        <p className={`text-sm ${isDarkMode ? 'text-text-secondary' : 'text-gray-600'}`}>
          Enter any public GitHub repository URL to clone and analyze it
        </p>
      </div>
      
      <div className="space-y-4">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Globe className={`absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 ${isDarkMode ? 'text-text-secondary' : 'text-gray-400'}`} />
            <input
              type="text"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="user/repo or https://github.com/user/repo"
              className={`w-full pl-10 pr-4 py-3 rounded-lg border text-sm font-medium ${
                isDarkMode 
                  ? 'bg-slate-800 border-slate-600 text-slate-100 placeholder-slate-400' 
                  : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
              } focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200`}
              disabled={isCloning}
            />
          </div>
          <button
            onClick={cloneRepo}
            disabled={isCloning || !repoUrl.trim()}
            className={`px-6 py-3 rounded-lg font-medium transition-colors ${
              isCloning || !repoUrl.trim()
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-primary text-white hover:bg-primary/90'
            }`}
          >
            {isCloning ? 'Cloning...' : 'Clone'}
          </button>
        </div>
        
        <div className={`text-xs ${isDarkMode ? 'text-text-secondary' : 'text-gray-500'}`}>
          <p className="mb-1">Supported formats:</p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">user/repo</code> (e.g., "facebook/react")</li>
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">https://github.com/user/repo</code></li>
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">https://github.com/user/repo.git</code></li>
          </ul>
        </div>
      </div>
    </div>
  )
}