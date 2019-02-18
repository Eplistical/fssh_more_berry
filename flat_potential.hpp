#ifndef _FLAT_POTENTIAL_HPP
#define _FLAT_POTENTIAL_HPP

#include <cstdlib>
#include <cmath>
#include <complex>
#include <algorithm>
#include <string>
#include "misc/crasher.hpp"
#include "misc/randomer.hpp"
#include "misc/vector.hpp"
#include "misc/matrixop.hpp"
#include "boost/math/special_functions/erf.hpp"

namespace {
    using std::vector;
    using std::complex;

    double A = 0.10;
    double B = 3.0;
    double W = 0.3;

    void output_potential_param() {
        /*
         * output potential paramters
         */
        ioer::info("Potential parameters: ", 
                    " A = ", A,
                    " B = ", B,
                    " W = ", W);
    }

    double cal_theta(const vector<double>& r) {
        /*
         * helper
         */
        const double x = r[0];
        return 0.5 * M_PI * (boost::math::erf(B * x) + 1);
    }

    vector<double> cal_nablatheta(const vector<double>& r) {
        /*
         * helper
         */
        const double x = r[0];
        vector<double> nablatheta(r.size(), 0.0);
        nablatheta[0] = std::sqrt(M_PI) * B * std::exp(-B * B * x * x);
        return nablatheta;
    }

    double cal_phi(const vector<double>& r) {
        /*
         * helper
         */
        const double y = r[1];
        return W * y;
    }

    vector<double> cal_nablaphi(const vector<double>& r) {
        /*
         * helper
         */
        vector<double> nablaphi(r.size(), 0.0);
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
        const double theta = cal_theta(r);
        const complex<double> eip = exp(matrixop::IMAGIZ * cal_phi(r));
        const double CC = cos(theta);
        const double SS = sin(theta);

        vector< complex<double> > H(4, 0.0);
        H[0+0*2] = -CC;
        H[1+1*2] = CC;
        H[0+1*2] = SS * eip;
        H[1+0*2] = conj(H[0+1*2]);
        H *= A;

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

        const double theta = cal_theta(r);
        const double phi = cal_phi(r);
        const vector<double> nablatheta = cal_nablatheta(r);
        const vector<double> nablaphi = cal_nablaphi(r);
        const complex<double> eip = exp(matrixop::IMAGIZ * phi);
        const double CC = cos(theta);
        const double SS = sin(theta);

        vector< vector< complex<double> > > nablaH(ndim);

        for (int ix(0); ix < ndim; ++ix) {
            vector< complex<double> >& nablaH_ix = nablaH[ix];
            nablaH_ix.resize(ndim * ndim);

            nablaH_ix[0+0*ndim] = SS * nablatheta[ix];
            nablaH_ix[1+1*ndim] = -SS * nablatheta[ix];
            nablaH_ix[0+1*ndim] = eip * (CC * nablatheta[ix] + matrixop::IMAGIZ * SS * nablaphi[ix]);
            nablaH_ix[1+0*ndim] = conj(nablaH_ix[0+1*ndim]);

            nablaH_ix *= A;
        }

        return nablaH;
    }

    void cal_info_nume(const vector<double>& r,
            vector<double>& eva, 
            vector< vector< complex<double> > >& dc,
            vector< vector<double> >& F,
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
                    F[ix][j+k*edim] = -dc[ix][j+k*edim].real();
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
