import React, { useEffect } from 'react';
import { useProjectStore } from '../lib/projectStore';
import { ChevronDown, Folder, Clock, HardDrive, Home } from 'lucide-react';

export const ProjectSelector: React.FC = () => {
  const { 
    projects, 
    currentProject, 
    setCurrentProject, 
    fetchProjects, 
    fetchFileTree,
    loading,
    error 
  } = useProjectStore();

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  const handleProjectChange = async (projectName: string) => {
    setCurrentProject(projectName);
    await fetchFileTree(projectName);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  if (loading) {
    return (
      <div className="p-4 border-b border-gray-200">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 border-b border-gray-200">
        <div className="text-red-500 text-sm">
          Error loading projects: {error}
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 border-b border-gray-200 bg-white">
      <div className="mb-3">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Projects</h3>
        {projects.length === 0 ? (
          <div className="text-gray-500 text-sm">
            No projects found. Clone a repository to get started.
          </div>
        ) : (
          <div className="space-y-2">
            {projects.map((project) => (
              <div
                key={project.name}
                className={`
                  p-3 rounded-lg border cursor-pointer transition-all duration-200
                  ${currentProject === project.name 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }
                `}
                onClick={() => handleProjectChange(project.name)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {project.is_workspace_root ? (
                      <Home className="w-4 h-4 text-orange-500" />
                    ) : (
                      <Folder className="w-4 h-4 text-blue-500" />
                    )}
                    <span className="font-medium text-sm text-gray-900">
                      {project.is_workspace_root ? 'Workspace Files' : project.name}
                    </span>
                  </div>
                  {currentProject === project.name && (
                    <ChevronDown className="w-4 h-4 text-blue-500" />
                  )}
                </div>
                
                <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
                  <div className="flex items-center space-x-1">
                    <Clock className="w-3 h-3" />
                    <span>{formatDate(project.modified)}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <HardDrive className="w-3 h-3" />
                    <span>{formatFileSize(project.size)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}; 