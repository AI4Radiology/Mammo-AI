import { Card } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { ExternalLink, Users, GraduationCap, UserCheck, Github } from 'lucide-react';
export const Footer = () => {
  return <footer className="mt-16 py-12 bg-card border-t border-border">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="grid md:grid-cols-2 gap-8">
          {/* Credits Section */}
          <Card className="p-6">
            <h3 className="text-xl font-semibold text-foreground mb-6 flex items-center gap-2">
              <Users className="h-5 w-5 text-primary" />
              Equipo del Proyecto
            </h3>
            
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-foreground mb-2 flex items-center gap-2">
                  <GraduationCap className="h-4 w-4 text-primary" />
                  Estudiantes
                </h4>
                <ul className="text-sm text-muted-foreground space-y-1 ml-6">
                  <li>Andrés Camilo Romero Ruiz</li>
                  <li>Juan Camilo Salazar Quintero</li>
                  <li>Brayan Steven Ortega García</li>
                </ul>
              </div>

              <Separator />

              <div>
                <h4 className="font-medium text-foreground mb-2 flex items-center gap-2">
                  <UserCheck className="h-4 w-4 text-primary" />
                  Tutores
                </h4>
                <ul className="text-sm text-muted-foreground space-y-1 ml-6">
                  <li>Ángela Villota</li>
                  <li>Aníbal Sosa</li>
                  <li>Andrés Aristizábal</li>
                </ul>
              </div>

              <Separator />

              <div>
                <h4 className="font-medium text-foreground mb-2">
                  Asesor Médico
                </h4>
                <p className="text-sm text-muted-foreground ml-6">
                  Juan Felipe Orejuela
                  <span className="block text-xs">Radiólogo, Fundación Valle del Lili</span>
                </p>
              </div>
            </div>
          </Card>

          {/* Information Section */}
          <Card className="p-6">
            <h3 className="text-xl font-semibold text-foreground mb-6">
              Información del Proyecto
            </h3>
            
            <div className="space-y-4">

                <div>
                  <h4 className="font-medium text-foreground mb-2">
                    ¿Cómo lo Hacemos?
                  </h4>
                  <p className="text-sm text-muted-foreground ml-6">
                    En el repositorio podrás encontrar el código fuente del proyecto, 
                    incluyendo la extracción de radiomicos, el modelo de inteligencia artificial, 
                    el código de entrenamiento, validación, la interfaz web, el ETL con watchdog
                    y la documentación técnica. Te invitamos a explorar el código, y si eres otro 
                    estudiante a contribuir con mejoras y compartir tus ideas para seguir avanzando
                    con esta iniciativa
                  </p>
                </div>

                <Separator />

                <div>
                  <h4 className="font-medium text-foreground mb-2">
                    No Aceptamos
                  </h4>
                  <p className="text-sm text-muted-foreground ml-6">
                    Mamografías con implantes mamarios, mamografías de pacientes masculinos o con vista 
                    oblicuolateral. Esto debido a que no forman parte del conjunto de datos con el que 
                    se entrenó el modelo de inteligencia artificial. Por lo que su uso podría llevar a 
                    resultados inexactos o erróneos.
                  </p>
                </div>
            </div>
          </Card>
        </div>

        <Separator className="my-8" />
        
        <div className="text-center text-sm text-muted-foreground space-y-3">
          <p>Universidad ICESI & Fundación Valle del Lili - {new Date().getFullYear()}</p>
          <a 
            href="https://github.com/BrayanOrteg/proyecto_de_grado" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-muted-foreground hover:text-primary transition-colors"
          >
            <Github className="h-5 w-5" />
            <span>Ver repositorio en GitHub</span>
          </a>
        </div>
      </div>
    </footer>;
};