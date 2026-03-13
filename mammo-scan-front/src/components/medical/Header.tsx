import IcesiLogo from '@/assets/ICESI_logo.svg';
import ValleLiliLogo from '@/assets/valle_del_lili.svg';

export const Header = () => {
  return (
    <header className="bg-card border-b border-border py-6">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="grid grid-cols-3 items-center">
          {/* Left Logo */}
          <div className="flex justify-start">
            <img 
              src={IcesiLogo} 
              alt="Logo Universidad ICESI" 
              className="h-20 w-auto"
            />
          </div>

          {/* Center Title */}
          <div className="text-center">
            <h1 className="text-2xl md:text-3xl font-bold text-foreground">
              Proyecto de Grado
            </h1>
            <h2 className="text-lg md:text-xl text-muted-foreground mt-1">
              Clasificación Automática de Mamografías
            </h2>
          </div>

          {/* Right Logo */}
          <div className="flex justify-end">
            <img 
              src={ValleLiliLogo} 
              alt="Logo Fundación Valle del Lili" 
              className="h-14 w-auto"
            />
          </div>
        </div>
      </div>
    </header>
  );
};
