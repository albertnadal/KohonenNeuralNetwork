#include <math.h>

const int huecolors=360;

typedef struct RGB
{
  unsigned char r;
  unsigned char g;
  unsigned char b;
} RGB;

RGB hue2rgb[huecolors];

void set_hue2rgb_channels(int idx, int r, int g, int b, double I)
{
  if (idx<0) return;
  if (idx>=huecolors) return;

/*
  const double target=0.85;
  I=1-I;
  I=target+I*(1-target);
*/
  I=1;

  hue2rgb[idx].r=r*I;
  hue2rgb[idx].g=g*I;
  hue2rgb[idx].b=b*I;
}

void init_hue2rgb()
{
  int s1=0;
  int s2=huecolors/3;
  int s3=2*huecolors/3;
  int s4=huecolors;
  int ss=1+(huecolors/3);
  for(int i = 0; i <= ss/2; i++)
    {
      double a=sin(i*M_PI/ss);
      int S=255.0*a;
      double j=2.0*(double)i/(double)ss;
      set_hue2rgb_channels(s1+i,255,S,0,j);
      set_hue2rgb_channels(s2+i,0,255,S,j);
      set_hue2rgb_channels(s3+i,S,0,255,j);
      set_hue2rgb_channels(s2-i,S,255,0,j);
      set_hue2rgb_channels(s3-i,0,S,255,j);
      set_hue2rgb_channels(s4-i,255,0,S,j);
    }
}

RGB* create_color_huebar(int bar_length)
{

   RGB *bar = (RGB *)malloc(sizeof(RGB) * bar_length);
   int sx = bar_length; //255

   init_hue2rgb();
   for(int x=0; x < sx; x++)
   {
       int i=(x*huecolors)/sx;
       int r=hue2rgb[i].r;
       int g=hue2rgb[i].g;
       int b=hue2rgb[i].b;
/*
       r*=195;
       r/=255;
       g*=175;
       g/=255;
*/
       bar[x].r = r;
       bar[x].g = g;
       bar[x].b = b;
   }

   return bar;
}
