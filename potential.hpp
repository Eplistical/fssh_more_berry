#ifndef _POTENTIAL_HPP
#define _POTENTIAL_HPP

#include <cstdlib>
#include <cmath>
#include <complex>
#include <algorithm>
#include <string>
#include "misc/crasher.hpp"
#include "misc/randomer.hpp"
#include "misc/vector.hpp"
#include "misc/matrixop.hpp"
#include "misc/ioer.hpp"

namespace {
    using std::vector;
    using std::complex;

    double W = 0.01;

    void output_potential_param() {
        /*
         * output potential paramters
         */
        ioer::info("# Potential parameters: ", 
                    " W = ", W);
    }

    void set_potenial_params(const std::vector<double>& params) {
        misc::crasher::confirm(params.size() >= 1, 
                "set_potenial_params: potential paramter vector size must be >= 1");
        W = params[0];
    }

    complex<double> cal_phi(const vector<double>& r) {
        /*
         * helper
         */
        return W * r[1];
    }

    vector< complex<double> > cal_nablaphi(const vector<double>& r) {
        /*
         * helper
         */
        vector< complex<double> > nablaphi(r.size(), 0.0);
        nablaphi[1] = W;
        return nablaphi;
    }

    vector< complex<double> > cal_H(const vector<double>& r) {
        /*
         * input: position vector r
         *
         * return: complex Hamiltonian H(r)
         *
         */
        const double x = r[0];
        const double y = r[1];
        const complex<double> eip = exp(matrixop::IMAGIZ * cal_phi(r));

        vector< complex<double> > H(4, 0.0);
        H[0+0*2] = x*x + y*y - x;
        H[1+1*2] = x*x + y*y + x;
        H[0+1*2] = y * eip;
        H[1+0*2] = conj(H[0+1*2]);

        return H;
    }

    vector< vector< complex<double> > > cal_nablaH(const vector<double>& r) {
        /*
         * input: position vector r
         *
         * return: gradiant of complex Hamiltonian (Hx, Hy, ...)
         *
         */
        const int ndim = r.size();

        const double x = r[0];
        const double y = r[1];
        const complex<double> eip = exp(matrixop::IMAGIZ * cal_phi(r));
        const vector< complex<double> > nablaphi = cal_nablaphi(r);

        vector< vector< complex<double> > > nablaH(ndim);
        vector< complex<double> >& Hx = nablaH[0];
        vector< complex<double> >& Hy = nablaH[1];
        
        Hx.resize(4);
        Hx[0+0*2] = 2 * x - 1;
        Hx[1+1*2] = 2 * x + 1;
        Hx[0+1*2] = y * eip * matrixop::IMAGIZ * nablaphi[0];
        Hx[1+0*2] = conj(Hx[0+1*2]);

        Hy.resize(4);
        Hy[0+0*2] = 2 * y;
        Hy[1+1*2] = 2 * y;
        Hy[0+1*2] = eip * (1.0 + matrixop::IMAGIZ * nablaphi[1]);
        Hy[1+0*2] = conj(Hy[0+1*2]);

        // other dimensions
        for (int ix(2); ix < ndim); ++ix) {
            nablaH[ix].resize(4);
        }

        return nablaH;
    }

    void cal_info_nume(const vector<double>& r,
            vector<double>& eva, 
            vector< vector< complex<double> > >& dc,
            vector< vector< complex<double> > >& F,
            vector< complex<double> >& lastevt)
    {
        /*
         * input:   position vector r
         *
         * output:  eigenvalues
         *          (phase aligned) derivative coupling matrix
         *          force matrix
         *
         * in/out:  last step eigenvectors lastevt (can be a empty vector)
         *              on exit, lastevt is replaced by current step evt
         *
         * -- calculation is performed in the numerical way
         */

        const vector< complex<double> > H = cal_H(r);
        const int ndim = r.size();
        const int edim = static_cast<int>(std::sqrt(static_cast<double>(H.size())));
        vector< complex<double> > evt;
        matrixop::hdiag(cal_H(r), eva, evt);

        // correct phase
        if (not lastevt.empty()) {
            auto tmp = matrixop::matCmat(lastevt, evt, edim);
            for (int j = 0; j < edim; ++j) {
                complex<double> eip = tmp[j+j*edim] / abs(tmp[j+j*edim]);
                for (int k = 0; k < edim; ++k) {
                    evt[k+j*edim] /= eip;
                }
            }
        }

        // F, dc
        const vector< vector< complex<double> > > nablaH = cal_nablaH(r);
        dc.resize(ndim);
        F.resize(ndim);

        for (int ix = 0; ix < ndim; ++ix) {
            dc[ix] = matrixop::matCmatmat(evt, nablaH[ix], evt, edim, edim);
            F[ix].assign(edim * edim, 0.0);

            for (int j = 0; j < edim; ++j) {
                for (int k = 0; k < edim; ++k) {
                    F[ix][j+k*edim] = -dc[ix][j+k*edim];
                    if (j == k) {
                        dc[ix][j+k*edim] = 0.0;
                    }
                    else {
                        dc[ix][j+k*edim] /= (eva[k] - eva[j]);
                    }
                }
            }
        }

        // save evt to lastevt
        lastevt = std::move(evt);
    }
};

#endif
