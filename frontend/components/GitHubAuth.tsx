'use client'

import React, { useState, useEffect } from 'react'
import { Github, ExternalLink, Plus, Globe } from 'lucide-react'
import { useThemeStore } from '../lib/store'

interface GitHubAuthProps {
  onAuthSuccess: (session: string) => void
}

interface Repository {
  name: string
  full_name: string
  description: string
  private: boolean
  clone_url: string
  updated_at: string
}

export default function GitHubAuth({ onAuthSuccess }: GitHubAuthProps) {
  const [isAuthenticating, setIsAuthenticating] = useState(false)
  const [repos, setRepos] = useState<Repository[]>([])
  const [loading, setLoading] = useState(false)
  const { isDarkMode } = useThemeStore()
  const [anyRepoUrl, setAnyRepoUrl] = useState('')
  const [isCloning, setIsCloning] = useState(false)
  const [showAnyRepoInput, setShowAnyRepoInput] = useState(false)

  useEffect(() => {
    // Check for auth success in URL params
    const urlParams = new URLSearchParams(window.location.search)
    const authStatus = urlParams.get('auth')
    if (authStatus === 'success') {
      // Clean URL
      window.history.replaceState({}, '', window.location.pathname)
      loadRepos()
      onAuthSuccess('ok')
    }
  }, [onAuthSuccess])

  const startAuth = async () => {
    setIsAuthenticating(true)
    const authUrl = 'http://localhost:8000/auth/login/github'
    const popup = window.open(authUrl, 'github-auth', 'width=600,height=700')
    const timer = setInterval(() => {
      if (popup?.closed) {
        clearInterval(timer)
        setIsAuthenticating(false)
        loadRepos()
      }
    }, 1000)
  }

  const loadRepos = async () => {
    setLoading(true)
    try {
      const r = await fetch('http://localhost:8000/auth/repos', { credentials: 'include' })
      if (r.ok) {
        setRepos(await r.json())
      }
    } catch (error) {
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  const clone = async (repo: Repository) => {
    await fetch('http://localhost:8000/auth/clone', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(repo)
    })
    alert(`Cloning ${repo.name}...`)
  }

  const cloneAnyRepo = async () => {
    if (!anyRepoUrl.trim()) return
    
    setIsCloning(true)
    try {
      const response = await fetch('http://localhost:8000/projects/clone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: anyRepoUrl.trim() })
      })
      
      const result = await response.json()
      
      if (response.ok) {
        alert(`Repository cloned successfully as '${result.project_name}'!`)
        setAnyRepoUrl('')
        setShowAnyRepoInput(false)
        // Refresh projects if there's a callback
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

  if (repos.length) {
    return (
      <div className={`p-6 rounded-lg border ${isDarkMode ? 'bg-background-secondary border-border' : 'bg-white border-gray-200'}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-text-primary' : 'text-gray-900'}`}>Your Repositories</h3>
          <button
            onClick={() => setShowAnyRepoInput(!showAnyRepoInput)}
            className={`flex items-center gap-2 px-3 py-1 rounded-md text-sm ${isDarkMode ? 'bg-primary text-white hover:bg-primary/90' : 'bg-primary text-white hover:bg-primary/90'}`}
          >
            <Globe className="w-4 h-4" />
            Clone Any Repo
          </button>
        </div>
        
        {showAnyRepoInput && (
          <div className={`mb-4 p-3 rounded border ${isDarkMode ? 'bg-background border-border' : 'bg-gray-50 border-gray-200'}`}>
            <div className="flex gap-2">
              <input
                type="text"
                value={anyRepoUrl}
                onChange={(e) => setAnyRepoUrl(e.target.value)}
                placeholder="user/repo or https://github.com/user/repo"
                className={`flex-1 px-3 py-2 rounded border text-sm ${isDarkMode ? 'bg-background-secondary border-border text-text-primary' : 'bg-white border-gray-300 text-gray-900'}`}
                onKeyPress={(e) => e.key === 'Enter' && cloneAnyRepo()}
                disabled={isCloning}
              />
              <button
                onClick={cloneAnyRepo}
                disabled={isCloning || !anyRepoUrl.trim()}
                className={`px-3 py-2 rounded text-sm font-medium ${isCloning || !anyRepoUrl.trim() ? 'bg-gray-300 text-gray-500' : 'bg-primary text-white hover:bg-primary/90'}`}
              >
                {isCloning ? 'Cloning...' : 'Clone'}
              </button>
            </div>
            <p className={`text-xs mt-1 ${isDarkMode ? 'text-text-secondary' : 'text-gray-500'}`}>
              Enter repository URL or user/repo format (e.g., "facebook/react")
            </p>
          </div>
        )}
        
        {loading ? 'Loading…' : (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {repos.map(repo => (
              <div key={repo.full_name} className={`p-3 rounded border cursor-pointer hover:border-primary`} onClick={() => clone(repo)}>
                <div className="flex items-center justify-between">
                  <span>{repo.name}</span>
                  <ExternalLink className="w-4 h-4 text-primary" />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className={`p-6 rounded-lg border text-center ${isDarkMode ? 'bg-background-secondary border-border' : 'bg-white border-gray-200'}`}>
      <Github className="w-12 h-12 mx-auto mb-4 text-primary" />
      <h3 className={`text-lg font-semibold mb-2 ${isDarkMode ? 'text-text-primary' : 'text-gray-900'}`}>Connect to GitHub</h3>
      <p className={`text-sm mb-4 ${isDarkMode ? 'text-text-secondary' : 'text-gray-600'}`}>Clone and work on your GitHub repositories with CodeWise</p>
      <button onClick={startAuth} disabled={isAuthenticating} className={`px-4 py-2 rounded-lg font-medium ${isAuthenticating ? 'bg-gray-300' : 'bg-primary text-white hover:bg-primary/90'}`}>
        {isAuthenticating ? 'Authenticating…' : 'Connect with GitHub'}
      </button>
    </div>
  )
} 