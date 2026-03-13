import { useState } from 'react';
import { ChevronLeft, ChevronRight, Check } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface MaskCarouselProps {
  masks: string[];
  onMaskSelect: (mask: string) => void;
  className?: string;
}

export const MaskCarousel = ({ masks, onMaskSelect, className = "" }: MaskCarouselProps) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedMask, setSelectedMask] = useState<string | null>(null);

  const nextMask = () => {
    setCurrentIndex((prev) => (prev + 1) % masks.length);
  };

  const prevMask = () => {
    setCurrentIndex((prev) => (prev - 1 + masks.length) % masks.length);
  };

  const selectMask = (mask: string) => {
    setSelectedMask(mask);
    onMaskSelect(mask);
  };

  // Generate mock mask visualization
  const generateMaskPattern = (index: number) => {
    const patterns = [
      'M 50 10 L 90 50 L 50 90 L 10 50 Z',
      'M 20 20 Q 50 10 80 20 Q 90 50 80 80 Q 50 90 20 80 Q 10 50 20 20 Z',
      'M 30 20 L 70 20 L 80 40 L 70 80 L 30 80 L 20 40 Z',
      'M 50 15 L 85 35 L 85 65 L 50 85 L 15 65 L 15 35 Z',
      'M 40 10 L 60 10 L 90 40 L 60 90 L 40 90 L 10 40 Z'
    ];
    return patterns[index % patterns.length];
  };

  return (
    <Card className={`p-8 ${className}`}>
      <div className="space-y-6">
        <div className="text-center">
          <h3 className="text-xl font-semibold text-foreground mb-2">
            Selección de Máscara
          </h3>
          <p className="text-muted-foreground">
            Revisa las máscaras generadas y selecciona la más adecuada
          </p>
        </div>

        <div className="relative">
          {/* Mask visualization */}
          <div className="bg-muted rounded-2xl p-8 mb-6 flex items-center justify-center min-h-[300px]">
            <svg viewBox="0 0 100 100" className="w-48 h-48">
              <rect width="100" height="100" fill="hsl(var(--muted))" rx="8" />
              <path
                d={generateMaskPattern(currentIndex)}
                fill="hsl(var(--primary))"
                fillOpacity="0.7"
                stroke="hsl(var(--primary))"
                strokeWidth="2"
              />
            </svg>
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between mb-6">
            <Button
              variant="outline"
              size="sm"
              onClick={prevMask}
              disabled={masks.length <= 1}
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>

            <div className="text-center">
              <p className="text-sm font-medium text-foreground">
                Máscara {currentIndex + 1} de {masks.length}
              </p>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={nextMask}
              disabled={masks.length <= 1}
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>

          {/* Selection button */}
          <div className="text-center">
            <Button
              onClick={() => selectMask(masks[currentIndex])}
              className={`transition-all duration-300 ${
                selectedMask === masks[currentIndex]
                  ? 'bg-success hover:bg-success/90 text-success-foreground'
                  : 'bg-primary hover:bg-primary-hover'
              }`}
            >
              {selectedMask === masks[currentIndex] ? (
                <>
                  <Check className="w-4 h-4 mr-2" />
                  Máscara Seleccionada
                </>
              ) : (
                'Seleccionar esta máscara'
              )}
            </Button>
          </div>

          {/* Mask indicators */}
          <div className="flex justify-center space-x-2 mt-4">
            {masks.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentIndex(index)}
                className={`w-3 h-3 rounded-full transition-all duration-200 ${
                  index === currentIndex
                    ? 'bg-primary scale-125'
                    : 'bg-muted-foreground/30 hover:bg-muted-foreground/50'
                }`}
              />
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
};