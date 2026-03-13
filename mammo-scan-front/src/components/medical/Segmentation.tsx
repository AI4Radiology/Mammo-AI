import { useState, useEffect } from 'react';
import { Loader2, Brain, Check } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

interface SegmentationProps {
  file: File | null;
  onSegmentationComplete: (masks: string[]) => void;
  className?: string;
}

export const Segmentation = ({ file, onSegmentationComplete, className = "" }: SegmentationProps) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<'idle' | 'processing' | 'complete'>('idle');

  useEffect(() => {
    if (file && status === 'idle') {
      startSegmentation();
    }
  }, [file, status]);

  const startSegmentation = async () => {
    setIsProcessing(true);
    setStatus('processing');
    setProgress(0);

    // Simulate segmentation process
    const progressSteps = [
      { progress: 20, message: "Cargando modelo Segment Anything..." },
      { progress: 40, message: "Analizando la imagen DICOM..." },
      { progress: 60, message: "Generando máscaras de segmentación..." },
      { progress: 80, message: "Optimizando resultados..." },
      { progress: 100, message: "Segmentación completada" }
    ];

    for (let i = 0; i < progressSteps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1500));
      setProgress(progressSteps[i].progress);
    }

    // Generate mock masks
    const mockMasks = [
      'mask_1', 'mask_2', 'mask_3', 'mask_4', 'mask_5'
    ];

    setIsProcessing(false);
    setStatus('complete');
    onSegmentationComplete(mockMasks);
  };

  return (
    <Card className={`p-8 transition-all duration-300 ${className}`}>
      <div className="text-center space-y-6">
        <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center transition-all duration-500 ${
          status === 'complete' 
            ? 'bg-success text-success-foreground' 
            : 'bg-primary text-primary-foreground'
        }`}>
          {status === 'complete' ? (
            <Check className="w-10 h-10" />
          ) : isProcessing ? (
            <Loader2 className="w-10 h-10 animate-spin" />
          ) : (
            <Brain className="w-10 h-10" />
          )}
        </div>

        <div className="space-y-3">
          <h3 className="text-xl font-semibold text-foreground">
            {status === 'complete' ? 'Segmentación Completada' : 'Segmentación con IA'}
          </h3>
          
          <p className="text-muted-foreground">
            {status === 'complete' 
              ? 'Las máscaras de segmentación han sido generadas exitosamente'
              : 'Utilizando Segment Anything para identificar regiones de interés'}
          </p>
        </div>

        {isProcessing && (
          <div className="space-y-4">
            <Progress value={progress} className="w-full max-w-md mx-auto" />
            <p className="text-sm text-muted-foreground">
              {progress}% completado
            </p>
          </div>
        )}

        {status === 'complete' && (
          <div className="mt-6 p-4 bg-success/10 rounded-lg border border-success/20">
            <p className="text-success font-medium">
              ✓ 5 máscaras de segmentación generadas
            </p>
          </div>
        )}
      </div>
    </Card>
  );
};