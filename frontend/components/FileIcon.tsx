import React from 'react';
import { 
  FaFolder, 
  FaFolderOpen, 
  FaFile, 
  FaFileCode, 
  FaFileImage, 
  FaFilePdf,
  FaFileAlt,
  FaFileArchive,
  FaFileAudio,
  FaFileVideo,
  FaJs,
  FaReact,
  FaPython,
  FaHtml5,
  FaCss3Alt,
  FaMarkdown,
  FaGitAlt,
  FaDocker,
  FaNodeJs
} from 'react-icons/fa';
import { 
  SiTypescript, 
  SiJson, 
  SiYaml,
  SiToml,
  SiSvelte,
  SiVuedotjs,
  SiNextdotjs,
  SiTailwindcss
} from 'react-icons/si';
import { VscJson, VscMarkdown } from 'react-icons/vsc';

interface FileIconProps {
  fileName: string;
  fileType: 'file' | 'directory';
  isExpanded?: boolean;
  className?: string;
}

const getFileIcon = (fileName: string, fileType: 'file' | 'directory', isExpanded?: boolean) => {
  if (fileType === 'directory') {
    return isExpanded ? <FaFolderOpen className="text-blue-500" /> : <FaFolder className="text-blue-500" />;
  }

  const extension = fileName.split('.').pop()?.toLowerCase();
  const fullName = fileName.toLowerCase();

  // Special files
  if (fullName === 'package.json') return <FaNodeJs className="text-green-600" />;
  if (fullName === 'dockerfile') return <FaDocker className="text-blue-600" />;
  if (fullName === '.gitignore' || fullName === '.gitattributes') return <FaGitAlt className="text-orange-600" />;
  if (fullName === 'readme.md') return <FaMarkdown className="text-blue-600" />;
  if (fullName === 'tailwind.config.js') return <SiTailwindcss className="text-teal-500" />;
  if (fullName === 'next.config.js') return <SiNextdotjs className="text-black" />;

  // By extension
  switch (extension) {
    case 'js':
      return <FaJs className="text-yellow-500" />;
    case 'jsx':
      return <FaReact className="text-blue-500" />;
    case 'ts':
      return <SiTypescript className="text-blue-600" />;
    case 'tsx':
      return <FaReact className="text-blue-500" />;
    case 'py':
      return <FaPython className="text-yellow-600" />;
    case 'html':
      return <FaHtml5 className="text-orange-600" />;
    case 'css':
      return <FaCss3Alt className="text-blue-500" />;
    case 'scss':
    case 'sass':
      return <FaCss3Alt className="text-pink-500" />;
    case 'json':
      return <VscJson className="text-yellow-600" />;
    case 'md':
    case 'markdown':
      return <VscMarkdown className="text-blue-600" />;
    case 'yaml':
    case 'yml':
      return <SiYaml className="text-red-500" />;
    case 'toml':
      return <SiToml className="text-gray-600" />;
    case 'svg':
      return <FaFileImage className="text-purple-500" />;
    case 'png':
    case 'jpg':
    case 'jpeg':
    case 'gif':
    case 'webp':
      return <FaFileImage className="text-green-500" />;
    case 'pdf':
      return <FaFilePdf className="text-red-600" />;
    case 'zip':
    case 'rar':
    case 'tar':
    case 'gz':
      return <FaFileArchive className="text-orange-500" />;
    case 'mp3':
    case 'wav':
    case 'flac':
      return <FaFileAudio className="text-purple-600" />;
    case 'mp4':
    case 'avi':
    case 'mkv':
    case 'mov':
      return <FaFileVideo className="text-red-500" />;
    case 'txt':
    case 'log':
      return <FaFileAlt className="text-gray-600" />;
    case 'vue':
      return <SiVuedotjs className="text-green-600" />;
    case 'svelte':
      return <SiSvelte className="text-orange-600" />;
    default:
      return <FaFileCode className="text-gray-500" />;
  }
};

export const FileIcon: React.FC<FileIconProps> = ({ fileName, fileType, isExpanded, className = "" }) => {
  const icon = getFileIcon(fileName, fileType, isExpanded);
  
  return (
    <span className={`inline-flex items-center justify-center w-4 h-4 ${className}`}>
      {icon}
    </span>
  );
}; 