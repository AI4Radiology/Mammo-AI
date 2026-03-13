import { useState } from 'react';
import { Header } from '@/components/medical/Header';
import { Footer } from '@/components/medical/Footer';
import { FileUpload } from '@/components/medical/FileUpload';
import { RadiomicsTable } from '@/components/medical/RadiomicsTable';
import { ClassificationResults } from '@/components/medical/ClassificationResults';
import { WorkflowProgress } from '@/components/medical/WorkflowProgress';
import { ClearButton } from '@/components/medical/ClearButton';

interface RadiomicsFeature {
  característica: string;
  valor: number;
}

const Index = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [radiomicsFeatures, setRadiomicsFeatures] = useState<RadiomicsFeature[]>([]);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setCurrentStep(0);
  };

  const handleFeaturesExtracted = (features: RadiomicsFeature[]) => {
    setRadiomicsFeatures(features);
    setCurrentStep(1);
  };

  const handleClassification = () => {
    setCurrentStep(2);
  };

  const handleClear = () => {
    setCurrentStep(0);
    setSelectedFile(null);
    setRadiomicsFeatures([]);
  };

  return (
    <div className="min-h-screen bg-gradient-background">
      <Header />
      
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Description */}
        <div className="text-center mb-8">
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Interfaz para cargar archivos DICOM, calcular características radiómicas 
            y clasificar la densidad mamaria utilizando machine learning.
          </p>
        </div>

        {/* Workflow Progress */}
        <div className="mb-12">
          <WorkflowProgress currentStep={currentStep} />
        </div>

        {/* Main Content */}
        <div className="space-y-8">
          {/* Step 1: File Upload */}
          <FileUpload
            onFileSelect={handleFileSelect}
            acceptedFormats={['.dicom', '.dcm']}
            label="Subir archivo DICOM"
            selectedFile={selectedFile}
            disabled={currentStep > 0}
            className="animate-slide-in"
          />

          {/* Step 2: Radiomics Extraction */}
          {selectedFile && (
            <RadiomicsTable
              file={selectedFile}
              onFeaturesExtracted={handleFeaturesExtracted}
              className="animate-slide-in"
            />
          )}

          {/* Step 3: Classification */}
          {radiomicsFeatures.length > 0 && (
            <ClassificationResults
              features={radiomicsFeatures}
              onClassify={handleClassification}
              className="animate-slide-in"
            />
          )}

          {/* Clear Button - Show when there's any progress */}
          {currentStep > 0 && (
            <ClearButton onClear={handleClear} />
          )}
        </div>
      </div>
      
      <Footer />
    </div>
  );
};

export default Index;
