import { useState } from "react";
import { Brain, Activity, BarChart3, AlertCircle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

interface RadiomicsFeature {
  característica: string;
  valor: number;
}

interface BackendClassificationResult {
  binary: {
    prediction: string; // "denso" | "no_denso"
    probabilities: Record<string, number>;
  };
  multiclass: {
    prediction: string; // "A" | "B" | "C" | "D"
    probabilities: Record<string, number>;
  };
}

interface ClassificationResultsProps {
  features: RadiomicsFeature[];
  onClassify: () => void;
  className?: string;
}

export const ClassificationResults = ({
  features,
  onClassify,
  className = "",
}: ClassificationResultsProps) => {
  const [isClassifying, setIsClassifying] = useState(false);
  const [results, setResults] = useState<BackendClassificationResult | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);

  const runClassification = async () => {
    setIsClassifying(true);
    setError(null);

    try {
      const payload = { columns: ["característica", "valor"], records: features };

      const response = await fetch("http://localhost:5000/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Error al clasificar");
      }

      const data = (await response.json()) as BackendClassificationResult;
      setResults(data);
      onClassify();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setIsClassifying(false);
    }
  };

  const getCategoryDescription = (cat: string) => {
    const desc: Record<string, string> = {
      A: "Tejido predominantemente graso",
      B: "Áreas dispersas de densidad fibroglandular",
      C: "Tejido heterogéneamente denso",
      D: "Tejido extremadamente denso",
    };
    return desc[cat] || "";
  };

  const getCategoryColor = (cat: string) => {
    const colors: Record<string, string> = {
      A: "bg-green-500",
      B: "bg-yellow-500",
      C: "bg-orange-500",
      D: "bg-red-500",
    };
    return colors[cat] || "bg-gray-400";
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {!results && (
        <Card className="p-6 text-center space-y-4">
          <div className="w-16 h-16 mx-auto bg-primary/10 rounded-full flex items-center justify-center">
            <Brain className="w-8 h-8 text-primary" />
          </div>
          <h3 className="text-lg font-semibold">Clasificación de Densidad Mamaria</h3>
          <p className="text-muted-foreground">
            Listo para clasificar usando las características radiómicas extraídas
          </p>

          {error && (
            <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-destructive mt-0.5" />
              <div>
                <h4 className="font-medium text-destructive">Error al clasificar</h4>
                <p className="text-sm text-destructive/80 mt-1">{error}</p>
              </div>
            </div>
          )}

          <Button
            onClick={runClassification}
            disabled={isClassifying || features.length === 0}
            className="bg-gradient-primary hover:opacity-90"
          >
            {isClassifying ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                Clasificando...
              </>
            ) : (
              <>
                <Activity className="w-4 h-4 mr-2" />
                Iniciar Clasificación
              </>
            )}
          </Button>
        </Card>
      )}

      {results && (
        <div className="space-y-6">
          {/* Binaria */}
          <Card className="p-6 space-y-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-primary/10 rounded-full flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-primary" />
              </div>
              <h3 className="text-lg font-semibold">Clasificación Binaria</h3>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Tipo:</span>
              <span
                className={`font-semibold px-3 py-1 rounded-full text-sm ${
                  results.binary.prediction === "denso"
                    ? "bg-orange-100 text-orange-800"
                    : "bg-green-100 text-green-800"
                }`}
              >
                {results.binary.prediction}
              </span>
            </div>
            <Progress
              value={
                results.binary.prediction === "denso"
                    ? results.binary.probabilities["denso"] * 100
                    : results.binary.probabilities["no_denso"] * 100
              }
              className="h-2"
            />
            <p className="text-sm">
              Probabilidad:{" "}
              {(results.binary.probabilities[results.binary.prediction] * 100).toFixed(1)}%
            </p>
          </Card>

          {/* Multiclase */}
          <Card className="p-6 space-y-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-accent/10 rounded-full flex items-center justify-center">
                <Activity className="w-5 h-5 text-accent" />
              </div>
              <h3 className="text-lg font-semibold">Clasificación Multiclase</h3>
            </div>

            <div className="text-center p-4 bg-muted/50 rounded-lg">
              <div className="text-2xl font-bold">
                Categoría {results.multiclass.prediction}
              </div>
              <div className="text-sm text-muted-foreground mt-1">
                {getCategoryDescription(results.multiclass.prediction)}
              </div>
            </div>

            {Object.entries(results.multiclass.probabilities).map(
              ([cat, prob]) => (
                <div key={cat} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="flex items-center space-x-2">
                      <div
                        className={`w-3 h-3 rounded-full ${getCategoryColor(cat)}`}
                      />
                      <span>Categoría {cat}</span>
                    </span>
                    <span className="font-medium">{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={prob * 100} className="h-1.5" />
                </div>
              )
            )}
          </Card>
        </div>
      )}
    </div>
  );
};