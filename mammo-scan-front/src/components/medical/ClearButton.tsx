import { Button } from '@/components/ui/button';
import { RotateCcw } from 'lucide-react';

interface ClearButtonProps {
  onClear: () => void;
  disabled?: boolean;
}

export const ClearButton = ({ onClear, disabled = false }: ClearButtonProps) => {
  return (
    <div className="flex justify-center mt-6">
      <Button
        variant="outline"
        size="lg"
        onClick={onClear}
        disabled={disabled}
        className="gap-2 animate-slide-in"
      >
        <RotateCcw className="h-4 w-4" />
        Limpiar y Reiniciar
      </Button>
    </div>
  );
};