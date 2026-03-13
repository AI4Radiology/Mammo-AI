import { useState, useCallback, useEffect } from 'react';
import { Upload, FileText, Check, Lock } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  acceptedFormats: string[];
  label: string;
  className?: string;
  disabled?: boolean;
  selectedFile?: File | null;
}

export const FileUpload = ({ onFileSelect, acceptedFormats, label, className = "", selectedFile: externalFile, disabled = false }: FileUploadProps) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Sincronizar con el estado externo
  useEffect(() => {
    if (externalFile === null) {
      setSelectedFile(null);
      // Limpiar el input file
      const fileInput = document.getElementById('file-upload') as HTMLInputElement;
      if (fileInput) {
        fileInput.value = '';
      }
    }
  }, [externalFile]);

  const handleDrag = useCallback((e: React.DragEvent) => {
    // Ignore drag if disabled
    if (disabled) return;

    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    // Ignore drop if disabled
    if (disabled) return;

    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    // If disabled, ignore selection and reset input
    if (disabled) {
      e.currentTarget.value = '';
      return;
    }

    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [onFileSelect, disabled]);

  return (
    <Card className={`p-8 transition-all duration-300 ${dragActive ? 'border-primary shadow-medical' : 'border-border'} ${className}`}>
      <div
        className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 ${
          disabled ? 'opacity-70 pointer-events-none' : '' } ${
          dragActive 
            ? 'border-primary bg-primary/5' 
            : selectedFile 
              ? 'border-success bg-success/5' 
              : 'border-muted-foreground/25 hover:border-primary/50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        data-disabled={disabled}
      >
        <div className="flex flex-col items-center space-y-4">
          <div className={`w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 ${
            selectedFile 
              ? 'bg-success text-success-foreground' 
              : 'bg-primary/10 text-primary'
          }`}>
            {selectedFile ? <Check className="w-8 h-8" /> : (disabled ? <Lock className="w-6 h-6" /> : <Upload className="w-8 h-8" />)}
          </div>
          
          <div className="space-y-2">
            <h3 className="text-lg font-semibold text-foreground">
              {selectedFile ? 'Archivo seleccionado' : label}
            </h3>
            
            {selectedFile ? (
              <div className="flex items-center space-x-2 text-success">
                <FileText className="w-4 h-4" />
                <span className="text-sm font-medium">{selectedFile.name}</span>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                {disabled ? 'Carga bloqueada hasta reiniciar. Presiona "Limpiar y Reiniciar" para desbloquear.' : 'Arrastra tu archivo aquí o haz clic para seleccionar'}
              </p>
            )}
          </div>

          <div className="text-xs text-muted-foreground">
            Formatos aceptados: {acceptedFormats.join(', ')}
          </div>

          <input
            type="file"
            id="file-upload"
            className="hidden"
            accept={acceptedFormats.join(',')}
            onChange={handleFileSelect}
            data-disabled={disabled}
            disabled={disabled}
          />
          
          <Button 
            variant="outline" 
            onClick={() => document.getElementById('file-upload')?.click()}
            className="mt-4"
            disabled={disabled}
          >
            {disabled ? (<><Lock className="h-4 w-4 mr-2" /> Carga bloqueada</>) : 'Seleccionar archivo'}
          </Button>
        </div>
      </div>
    </Card>
  );
};