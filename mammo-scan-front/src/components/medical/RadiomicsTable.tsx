import { useState, useEffect } from 'react';
import { Download, Calculator, Check, AlertCircle } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

interface RadiomicsFeature {
  característica: string;
  valor: number;
}

interface RadiomicsTableProps {
  file: File | null;
  onFeaturesExtracted: (features: RadiomicsFeature[]) => void;
  className?: string;
}

export const RadiomicsTable = ({ file, onFeaturesExtracted, className = "" }: RadiomicsTableProps) => {
  const [features, setFeatures] = useState<RadiomicsFeature[]>([]);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (file && !isComplete) {
      extractFeatures();
    } else if (!file) {
      // Resetear estado cuando no hay archivo
      setFeatures([]);
      setIsComplete(false);
      setError(null);
    }
  }, [file, isComplete]);

  const extractFeatures = async () => {
    if (!file) return;
    
    setIsExtracting(true);
    setError(null);
    
    try {
      // Crear FormData con el archivo DICOM
      const formData = new FormData();
      formData.append('dicom', file);

      // Llamar al endpoint /radiomics
      const response = await fetch('http://localhost:5000/radiomics', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Error al extraer características radiómicas');
      }

      const data = await response.json();
      
      // El backend retorna: { columns: ['característica', 'valor'], records: [{característica: '...', valor: ...}] }
      const extractedFeatures: RadiomicsFeature[] = data.records;

      setFeatures(extractedFeatures);
      setIsComplete(true);
      onFeaturesExtracted(extractedFeatures);
    } catch (err) {
      console.error('Error extrayendo características:', err);
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setIsExtracting(false);
    }
  };

  const downloadCSV = () => {
    const headers = ['Caracteristica', 'Valor'];
    const csvContent = [
      headers.join(','),
      ...features.map(f => [f.característica, f.valor.toString()].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'radiomics_features.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <Card className={`p-6 ${className}`}>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
              isComplete ? 'bg-success text-success-foreground' : 'bg-primary text-primary-foreground'
            }`}>
              {isComplete ? <Check className="w-5 h-5" /> : <Calculator className="w-5 h-5" />}
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">
                Características Radiómicas
              </h3>
              <p className="text-sm text-muted-foreground">
                {isExtracting ? 'Extrayendo características...' : 
                 isComplete ? 'Extracción completada' : 'Listo para extraer'}
              </p>
            </div>
          </div>

          {isComplete && (
            <Button onClick={downloadCSV} variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Descargar CSV
            </Button>
          )}
        </div>

        {isExtracting && (
          <div className="text-center py-8">
            <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-muted-foreground">Calculando características radiómicas...</p>
          </div>
        )}

        {error && (
          <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-medium text-destructive">Error al extraer características</h4>
              <p className="text-sm text-destructive/80 mt-1">{error}</p>
            </div>
          </div>
        )}

        {isComplete && features.length > 0 && (
          <div>
            <div className="mb-2 text-sm text-muted-foreground">
              Total de características: {features.length}
            </div>
            <div className="border rounded-lg overflow-hidden">
              <div className="max-h-96 overflow-y-auto">
                <Table>
                  <TableHeader className="sticky top-0 bg-background z-10">
                    <TableRow>
                      <TableHead className="bg-muted">Característica</TableHead>
                      <TableHead className="text-right bg-muted">Valor</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {features.map((feature, index) => (
                      <TableRow key={index} className="hover:bg-muted/50">
                        <TableCell className="font-mono font-medium">
                          {feature.característica}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {typeof feature.valor === 'number' ? feature.valor.toFixed(6) : feature.valor}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};