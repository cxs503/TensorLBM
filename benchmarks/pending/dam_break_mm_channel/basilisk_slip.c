/* Independent two-phase geometric VOF calculation on the free-slip domain.
 * Length unit a, velocity unit sqrt(2 g a), time unit sqrt(a/(2 g)).
 */
#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "tension.h"

static int cells;

static double crossing (const double * profile, int count)
{
  if (profile[count - 1] >= 0.5)
    return (double)count/cells;
  for (int j = count - 2; j >= 0; j--)
    if (profile[j] >= 0.5)
      return (j + 0.5 + (profile[j] - 0.5)/(profile[j] - profile[j + 1]))/cells;
  return 0.;
}

int main (int argc, char ** argv)
{
  cells = argc == 2 ? atoi (argv[1]) : 16;
  if (cells < 4 || (cells & (cells - 1)))
    return 2;
  dimensions (nx = 15, ny = 4);
  size (15.);
  origin (0., 0.);
  init_grid (15*cells);

  const double a_phys = 0.05715, g_phys = 9.81, rho_water = 998.;
  const double velocity = sqrt (2.*g_phys*a_phys);
  rho1 = 1.;
  rho2 = 1.2/rho_water;
  mu1 = 1e-6/(velocity*a_phys);
  mu2 = 1.8e-5/(rho_water*velocity*a_phys);
  f.sigma = 0.072/(rho_water*sq(velocity)*a_phys);
  CFL = 0.4;
  DT = 0.01;
  TOLERANCE = 1e-4;
  run();
}

event init (i = 0)
{
  fraction (f, min(1. - x, 2. - y));
  puts ("T,Z,H,V");
}

event acceleration (i++)
{
  face vector applied = a;
  foreach_face(y)
    applied.y[] -= 0.5;
}

event gauges (t = 0; t += 0.02; t <= 7.0)
{
  double bed[15*cells], wall[4*cells];
  double volume = 0.;
  foreach(serial) {
    if (y < Delta)
      bed[(int)(x*cells)] = f[];
    if (x < Delta)
      wall[(int)(y*cells)] = f[];
    volume += f[]*sq(Delta);
  }
  printf ("%.8g,%.10g,%.10g,%.16g\n", t, crossing (bed, 15*cells),
          crossing (wall, 4*cells)/2., volume);
  fflush (stdout);
}
