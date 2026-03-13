import { Check, Upload, Calculator, BarChart3 } from 'lucide-react';

interface WorkflowStep {
  id: string;
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  status: 'pending' | 'active' | 'completed';
}

interface WorkflowProgressProps {
  currentStep: number;
  className?: string;
}

export const WorkflowProgress = ({ currentStep, className = "" }: WorkflowProgressProps) => {
  const steps: WorkflowStep[] = [
    { id: 'upload', title: 'Carga de archivo', icon: Upload, status: 'pending' },
    { id: 'radiomics', title: 'Extracción radiómicas', icon: Calculator, status: 'pending' },
    { id: 'classification', title: 'Clasificación', icon: BarChart3, status: 'pending' },
  ];

  // Update step statuses based on current step
  const updatedSteps = steps.map((step, index) => ({
    ...step,
    status: index < currentStep 
      ? 'completed' as const
      : index === currentStep 
        ? 'active' as const 
        : 'pending' as const
  }));

  const getStepStyles = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-success text-success-foreground border-success shadow-glow';
      case 'active':
        return 'bg-primary text-primary-foreground border-primary shadow-medical animate-pulse-medical';
      default:
        return 'bg-muted text-muted-foreground border-border';
    }
  };

  return (
    <div className={`${className}`}>
      <div className="flex justify-between items-center relative">
        {/* Progress line */}
        <div className="absolute top-6 left-6 right-6 h-0.5 bg-border z-0">
          <div 
            className="h-full bg-primary transition-all duration-500 ease-out"
            style={{ width: `${(currentStep / (steps.length - 1)) * 100}%` }}
          />
        </div>

        {updatedSteps.map((step, index) => {
          const Icon = step.icon;
          return (
            <div key={step.id} className="flex flex-col items-center space-y-2 relative z-10">
              <div className={`w-12 h-12 rounded-full border-2 flex items-center justify-center transition-all duration-300 ${getStepStyles(step.status)}`}>
                {step.status === 'completed' ? (
                  <Check className="w-5 h-5" />
                ) : (
                  <Icon className="w-5 h-5" />
                )}
              </div>
              <span className={`text-sm font-medium text-center max-w-20 leading-tight ${
                step.status === 'active' 
                  ? 'text-primary' 
                  : step.status === 'completed' 
                    ? 'text-success' 
                    : 'text-muted-foreground'
              }`}>
                {step.title}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};