'use client'

import React, { useState, useRef } from 'react'
import { X, Folder, Upload, Loader2 } from 'lucide-react'
import { useThemeStore } from '../lib/store'
import { useProjectStore } from '../lib/projectStore'

interface ImportDialogProps {
  isOpen: boolean
  onClose: () => void
}

export default function ImportDialog({ isOpen, onClose }: ImportDialogProps) {
  const { isDarkMode } = useThemeStore()
  const { fetchProjects } = useProjectStore()
  const [isImporting, setIsImporting] = useState(false)
  const [importStatus, setImportStatus] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFolderSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files || files.length === 0) return

    // Get the first file to extract the folder path
    const firstFile = files[0]
    const webkitRelativePath = (firstFile as any).webkitRelativePath
    if (!webkitRelativePath) return

    // Extract project name from path (first directory)
    const projectName = webkitRelativePath.split('/')[0]
    
    try {
      setIsImporting(true)
      setImportStatus('Preparing project files...')

      // Create FormData with all selected files
      const formData = new FormData()
      formData.append('project_name', projectName)
      
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const relativePath = (file as any).webkitRelativePath
        formData.append('files', file, relativePath)
      }

      setImportStatus('Uploading and indexing project...')

      // Post to backend import endpoint
      const response = await fetch('http://localhost:8000/projects/import', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Import failed: ${response.statusText}`)
      }

      const result = await response.json()
      setImportStatus(`Successfully imported ${projectName}!`)
      
      // Refresh projects list
      await fetchProjects()
      
      // Close dialog after a brief success message
      setTimeout(() => {
        onClose()
        setImportStatus('')
        setIsImporting(false)
      }, 2000)

    } catch (error: any) {
      console.error('Import error:', error)
      setImportStatus(`Import failed: ${error.message}`)
      setTimeout(() => {
        setImportStatus('')
        setIsImporting(false)
      }, 3000)
    }
  }

  const handleImportClick = () => {
    fileInputRef.current?.click()
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className={`relative w-full max-w-md mx-4 rounded-lg shadow-xl ${
        isDarkMode ? 'bg-background-secondary' : 'bg-white'
      }`}>
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <h2 className={`text-xl font-semibold ${
            isDarkMode ? 'text-text-primary' : 'text-gray-900'
          }`}>
            Import Project
          </h2>
          <button
            onClick={onClose}
            disabled={isImporting}
            className={`p-1 rounded-lg transition-colors ${
              isDarkMode
                ? 'hover:bg-slate-600 text-text-secondary'
                : 'hover:bg-gray-100 text-gray-500'
            }`}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {!isImporting && !importStatus && (
            <div className="text-center">
              <div className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4 ${
                isDarkMode ? 'bg-slate-700' : 'bg-gray-100'
              }`}>
                <Folder className={`w-8 h-8 ${
                  isDarkMode ? 'text-text-secondary' : 'text-gray-500'
                }`} />
              </div>
              
              <h3 className={`text-lg font-medium mb-2 ${
                isDarkMode ? 'text-text-primary' : 'text-gray-900'
              }`}>
                Select Project Folder
              </h3>
              
              <p className={`text-sm mb-6 ${
                isDarkMode ? 'text-text-secondary' : 'text-gray-600'
              }`}>
                Choose a folder containing your project files. CodeWise will index all supported files for analysis.
              </p>

              <button
                onClick={handleImportClick}
                className="flex items-center gap-2 mx-auto px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors"
              >
                <Upload className="w-4 h-4" />
                Choose Folder
              </button>

              {/* Hidden file input for folder selection */}
              <input
                ref={fileInputRef}
                type="file"
                /* @ts-ignore - webkitdirectory is a valid HTML attribute */
                webkitdirectory=""
                multiple
                style={{ display: 'none' }}
                onChange={handleFolderSelect}
              />
            </div>
          )}

          {(isImporting || importStatus) && (
            <div className="text-center">
              <div className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4 ${
                isDarkMode ? 'bg-slate-700' : 'bg-gray-100'
              }`}>
                {isImporting ? (
                  <Loader2 className={`w-8 h-8 animate-spin ${
                    isDarkMode ? 'text-accent' : 'text-primary'
                  }`} />
                ) : (
                  <Folder className={`w-8 h-8 ${
                    isDarkMode ? 'text-green-400' : 'text-green-600'
                  }`} />
                )}
              </div>
              
              <p className={`text-sm ${
                isDarkMode ? 'text-text-primary' : 'text-gray-900'
              }`}>
                {importStatus}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}