export interface Project {
  name: string;
  path: string;
  modified: string;
  size: number;
  is_workspace_root?: boolean;
}

export interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  modified: string;
  size?: number;
  mime_type?: string;
  children?: FileNode[];
}

export interface FileContent {
  content: string | null;
  path: string;
  name: string;
  size: number;
  modified: string;
  mime_type: string;
  encoding: string;
  is_binary?: boolean;
}

export interface ProjectState {
  projects: Project[];
  currentProject: string | null;
  fileTree: FileNode | null;
  openFiles: FileContent[];
  activeFile: string | null;
  loading: boolean;
  error: string | null;
} 