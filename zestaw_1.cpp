#include <iostream>
#include <math.h>
using namespace std;

//miara Legesgue'a przedziału (a,b)
double v(double a, double b) { return a<b ? b-a : 0; }

// bierze przedziały (x11,x12) i (x21,x22), odwraca je, jeżeli ich
// miara jest równa 0, a następnie liczy miary przedziałów (x11,x22)
// i (x21,x12) , a następnie zwraca miarę najmniejszego z nich.
// to działa bo |A \cap B| <= |A| i |A \cap B| <= |B|, a inne
// przedziały niż A \cap B i A \cup B nie powstaną.
double intersection(double x11, double x12, double x21, double x22){
	double buff;
	if (x12 < x11) { buff = x12; x12=x11; x11=buff; }
	if (x22 < x21) { buff = x22; x22=x21; x21=buff; }
	
	double a1 = v(x11, x12);
	double a2 = v(x21, x22);
	double a3 = v(x21, x12);
	double a4 = v(x11, x22);
	
	if (a1<a2 and a1<a3 and a1<a4) return a1;
	if (a2<a1 and a2<a3 and a2<a4) return a2;
	if (a3<a2 and a3<a1 and a3<a4) return a3;
	if (a4<a2 and a4<a3 and a4<a1) return a4;
	
	//stary kod
	//~ // x1 - przedział 1, x2 - przedział 2
	//~ double buff;
	//~ //naprawienie kolejności punktów (powinno być x11 < x12)
	//~ if (x11 > x12){
		//~ buff = x11; x11 = x12; x12 = buff;
	//~ }
	//~ if (x21 > x22){
		//~ buff = x21; x21 = x22; x22 = buff;
	//~ }
	//~ //zamiana przedziałów tak, zeby przedział 1 miał "początek" "po lewej"
	//~ if (x21 < x11){
		//~ buff = x11; x11 = x21; x21 = buff;
		//~ buff = x12; x12 = x22; x22 = buff;
	//~ }
	//~ //właściwe sprawdzanie przecięcia
	//~ if (x21 > x12) // rozłączne
		//~ return 0.0;
	//~ if (x22 > x12)
		//~ return x21 - x12;
	//~ else // (x22 < x12)
		//~ return x22 - x12;
	//koniec starego kodu
}
int main(){
	double x11,x12,x21,x22;
	double y11,y12,y21,y22;
	cin >> x11 >> y11; // wsp. 1 pkt 1 prostokąta
	cin >> x12 >> y12; // wsp. 2 pkt 1 prostokąta
	cin >> x21 >> y21; // wsp. 1 pkt 2 prostokąta
	cin >> x22 >> y22; // wsp. 2 pkt 2 prostokąta
	
	double x = intersection(x11,x12,x21,x22);
	double y = intersection(y11,y12,y21,y22);
	
    if (x*y != 0)
    {
	    cout << "pole: " << x * y << endl;
	    cout << "obw: " << 2 * (x + y) << endl;
    } 
    else 
    { 
        cout<<"pole: 0"<<endl<<"obw: 0"<<endl;
    }
}
