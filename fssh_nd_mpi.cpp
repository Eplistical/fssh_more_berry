#include <cstdlib>
#include <cmath>
#include <complex>
#include <algorithm>
#include <string>
#include <utility>
#include "misc/fmtstring.hpp"
#include "misc/ioer.hpp"
#include "misc/crasher.hpp"
#include "misc/randomer.hpp"
#include "misc/vector.hpp"
#include "misc/timer.hpp"
#include "misc/matrixop.hpp"
#include "misc/MPIer.hpp"
#include "boost/numeric/odeint.hpp"
#include "boost/math/special_functions/erf.hpp"
#include "boost/program_options.hpp"
#include "ci_potential.hpp"
//#include "potential.hpp"
//#include "flat_potential.hpp"

enum {
    HOP_UP,
    HOP_DN,
    HOP_RJ,
    HOP_FR
};

using namespace std;
namespace po = boost::program_options;
using boost::numeric::odeint::runge_kutta4;
using boost::math::erf;
using state_t = vector< complex<double> >;

int ndim = 3;
int edim = 2;

vector<double> mass { 1000.0, 1000.0, 1000.0};
vector<double> init_r { -3.0, 0.0, 0.0};
vector<double> init_p { 30.0, 0.0, 0.0 };
vector<double> sigma_r { 0.5, 0.5, 0.0 };
vector<double> sigma_p { 1.0, 1.0, 0.0 };
vector<double> init_s { 0.0, 1.0 };

vector<double> potential_params;

int Nstep = 10000;
double dt = 0.1;
int output_step = 100;
int Ntraj = 2000;
int seed = 0;
string output_mod = "init_s";

double xwall_left = -10.0;
double xwall_right = 10.0;

vector<double> eva;
vector< vector< complex<double> > > dc;
vector< vector< complex<double> > > F;
vector< complex<double> > lastevt;

bool argparse(int argc, char** argv) 
{
    /*
     * parse input arguments
     */
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("Ntraj", po::value<int>(&Ntraj), "# traj")
        ("Nstep", po::value<int>(&Nstep), "# step")
        ("output_step", po::value<int>(&output_step), "# step for output")
        ("dt", po::value<double>(&dt), "single time step")
        ("mass", po::value< vector<double> >(&mass)->multitoken(), "mass vector")
        ("init_r", po::value< vector<double> >(&init_r)->multitoken(), "init_r vector")
        ("init_p", po::value< vector<double> >(&init_p)->multitoken(), "init_p vector")
        ("sigma_r", po::value< vector<double> >(&sigma_r)->multitoken(), "sigma_r vector")
        ("sigma_p", po::value< vector<double> >(&sigma_p)->multitoken(), "sigma_p vector")
        ("init_s", po::value< vector<double> >(&init_s)->multitoken(), "init_s vector")
        ("xwall_left", po::value<double>(&xwall_left), "x wall left")
        ("xwall_right", po::value<double>(&xwall_right), "x wall right")
        ("potential_params", po::value< vector<double> >(&potential_params)->multitoken(), "potential_params vector")
        ("seed", po::value<int>(&seed), "random seed")
        ("output_mod", po::value<string>(&output_mod), "output mode, init_s or init_px")
        ;
    po::variables_map vm; 
    //po::store(po::parse_command_line(argc, argv, desc), vm);
    po::store(parse_command_line(argc, argv, desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);
    po::notify(vm);    

    ndim = mass.size();
    edim = init_s.size();

    if (not potential_params.empty()) {
        set_potenial_params(potential_params);
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return false;
    }
    return true;
}

void init_state(state_t& state, const vector<double>& init_r, const vector<double>& init_p, 
        const vector<double>& mass, const vector<double> init_s) 
{
    /*
     * initialize state information for a single particle
     */

    // check
    const int ndim = mass.size();
    const int edim = init_s.size();
    misc::crasher::confirm(init_r.size() == ndim and init_p.size() == ndim,
                                "init_state: mass, init_r, init_p must have identical sizes");
    misc::crasher::confirm(std::all_of(init_s.begin(), init_s.end(), [](double si) { return si >= 0.0; }), 
                                "init_state: all init_s elements must be non-negative" );

    // state -> (r, p, c, s)
    const int Lstate = ndim * 2 + edim + 1; 
    state.resize(Lstate, matrixop::ZEROZ);

    // init nuclear DoF (r, p)
    for (int i(0); i < ndim; ++i) {
        state[i].real(randomer::normal(init_r[i], sigma_r[i]));

        if (sigma_p[i] == 0.0) {
            // if sigma = 0
            state[ndim + i].real(init_p[i]);
        }
        else if (i == 0) {
            // if x direction must be positive
            while (state[ndim + i].real() <= 0.0) {
                state[ndim + i].real( randomer::normal(init_p[i], sigma_p[i]) ); 
            }
        }
        else {
            state[ndim + i].real( randomer::normal(init_p[i], sigma_p[i]) ); 
        }
    }

    // init electronic DoF (c, s)
    vector<double> s_normalized = init_s / sum(init_s);
    for (int i(0); i < edim; ++i) {
        state[ndim * 2 + i] = sqrt(s_normalized[i]);
    }
    state[ndim * 2 + edim].real( randomer::discrete(s_normalized.begin(), s_normalized.end() ) );

}


bool check_end(const state_t& state) {
    /*
     * check whether a trajectory has left the reactive region
     */

    // extract information
    double x = state[0].real();
    double px = state[0 + ndim].real();
    if (x < xwall_left and px < 0.0) {
        return true;
    }
    else if (x > xwall_right and px > 0.0) {
        return true;
    }
    return false;
}

void VV_integrator(state_t& state, const vector<double>& mass, const double t, const double dt) 
{
    /*
     *  VV integrator
     */

    // check
    const int ndim = mass.size();
    const int edim = state.size() - 2 * ndim - 1;

    // extract information
    vector<double> r(ndim);
    vector<double> p(ndim);
    vector< complex<double> > c(edim);
    int s;
    for (int i(0); i < ndim; ++i) {
        r[i] = state[i].real();
        p[i] = state[ndim + i].real();
    }
    for (int i(0); i < edim; ++i) {
        c[i] = state[ndim * 2 + i];
    }
    s = static_cast<int>(state[ndim * 2 + edim].real());

    const double half_dt = 0.5 * dt;

    // nuclear part -- VV integrator
    vector<double> force(ndim);

    cal_info_nume(r, eva, dc, F, lastevt);

    complex<double> vdotdc(0.0, 0.0);
    for (int k(0); k < ndim; ++k) {
        vdotdc += p[k] / mass[k] * dc[k][(1-s)+s*edim];
    }

    for (int i(0); i < ndim; ++i) {
        force[i] = F[i][s + s * edim].real();
        // Berry force
        force[i] += 2 * (dc[i][s+(1-s)*edim] * vdotdc).imag();
    }
    p += half_dt * force;

    r += dt * p / mass;

    cal_info_nume(r, eva, dc, F, lastevt);

    vdotdc = complex<double>(0.0, 0.0);
    for (int k(0); k < ndim; ++k) {
        vdotdc += p[k] / mass[k] * dc[k][(1-s)+s*edim];
    }

    for (int i(0); i < ndim; ++i) {
        force[i] = F[i][s + s * edim].real();
    }
    p += half_dt * force;

    // electronic part -- RK4
    vector< complex<double> > rk4_mat(edim * edim, 0.0);
    for (int j(0); j < edim; ++j) {
        for (int k(0); k < edim; ++k) {
            for (int i(0); i < ndim; ++i) {
                rk4_mat[j+k*edim] -= p[i] * dc[i][j+k*edim] / mass[i];
            }
        }
    }
    for (int j(0); j < edim; ++j) {
        rk4_mat[j+j*edim] -= matrixop::IMAGIZ * eva[j];
    }

    vector< complex<double> > k1, k2, k3, k4;
    k1 = dt * matrixop::matmat(rk4_mat, c, edim);
    k2 = dt * matrixop::matmat(rk4_mat, c + 0.5 * k1, edim);
    k3 = dt * matrixop::matmat(rk4_mat, c + 0.5 * k2, edim);
    k4 = dt * matrixop::matmat(rk4_mat, c + k3, edim);
    c += 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    // replace r, p, c w/ t+dt results
    for (int i(0); i < ndim; ++i) {
        state[i].real(r[i]);
        state[ndim + i].real(p[i]);
    }
    for (int i(0); i < edim; ++i) {
        state[ndim * 2 + i] = c[i];
    }
}

void Boris_integrator(state_t& state, const vector<double>& mass, const double t, const double dt) 
{
    /*
     *  Boris integrator
     */

    // check
    const int ndim = mass.size();
    const int edim = state.size() - 2 * ndim - 1;
    misc::crasher::confirm(ndim == 3, "integrator: this integrator can only be apllied to ndim = 3");


    // extract information
    vector<double> r(ndim);
    vector<double> p(ndim);
    vector< complex<double> > c(edim);
    int s;
    for (int i(0); i < ndim; ++i) {
        r[i] = state[i].real();
        p[i] = state[ndim + i].real();
    }
    for (int i(0); i < edim; ++i) {
        c[i] = state[ndim * 2 + i];
    }
    s = static_cast<int>(state[ndim * 2 + edim].real());

    const double half_dt = 0.5 * dt;

    // nuclear part -- Boris integrator
    r += half_dt * p / mass;

    // calculte info 
    cal_info_nume(r, eva, dc, F, lastevt);
    vector<double> force(ndim);
    for (int i(0); i < ndim; ++i) {
        force[i] = F[i][s + s * edim].real();
    }

    p += half_dt * force;

    // calc w so that cross(p, w) = F^mag * 0.5 * dt
    const double w0 = 0.0, w1 = 0.0;
    double w2 = 0.0;
    for (int k(0); k < edim; ++k) {
        if (k != s) {
            w2 += (dc[0][s+k*edim] * dc[1][k+s*edim]).imag();
        }
    }
    w2 *= 2.0 / mass[2] * half_dt;

    // p -> (I + w) * inv(I - w) * p
    const double w00 = w0*w0, w11 = w1*w1, w22 = w2*w2;
    const double w01 = w0*w1, w02 = w0*w2, w12 = w1*w2;
    const double p0 = p[0], p1 = p[1], p2 = p[2];
    p[0] = (w00 - w11 - w22 + 1.0) * p0 + 2 * (w2 + w01) * p1 + 2 * (w02 - w1) * p2;
    p[1] = 2 * (w01 - w2) * p0 + (w00 + w11 - w22 + 1.0) * p1 + 2 * (w12 + w0) * p2;
    p[2] = 2 * (w1 + w02) * p0 + 2 * (w12 - w0) * p1 + (w00 - w11 + w22 + 1.0) * p2;
    p /= (w00 + w11 + w22 + 1.0);

    p += half_dt * force;
    r += half_dt * p / mass;

    // electronic part -- RK4
    vector< complex<double> > rk4_mat(edim * edim, 0.0);
    for (int j(0); j < edim; ++j) {
        for (int k(0); k < edim; ++k) {
            for (int i(0); i < ndim; ++i) {
                rk4_mat[j+k*edim] -= p[i] * dc[i][j+k*edim] / mass[i];
            }
        }
    }
    for (int j(0); j < edim; ++j) {
        rk4_mat[j+j*edim] -= matrixop::IMAGIZ * eva[j];
    }

    vector< complex<double> > k1, k2, k3, k4;
    k1 = dt * matrixop::matmat(rk4_mat, c, edim);
    k2 = dt * matrixop::matmat(rk4_mat, c + 0.5 * k1, edim);
    k3 = dt * matrixop::matmat(rk4_mat, c + 0.5 * k2, edim);
    k4 = dt * matrixop::matmat(rk4_mat, c + k3, edim);
    c += 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    // replace r, p, c w/ t+dt results
    for (int i(0); i < ndim; ++i) {
        state[i].real(r[i]);
        state[ndim + i].real(p[i]);
    }
    for (int i(0); i < edim; ++i) {
        state[ndim * 2 + i] = c[i];
    }

    // calc info again for correction
    cal_info_nume(r, eva, dc, F, lastevt);
}

template <typename ...Params>
void integrator(Params&&... params)
{
    /*
     * integrator interface
     */
    Boris_integrator(std::forward<Params>(params)...);
}

int hopper(state_t& state, const vector<double>& mass) 
{
    /*
     * 2-state hopper 
     */

    // check
    const int ndim = mass.size();
    const int edim = state.size() - 2 * ndim - 1;
    misc::crasher::confirm(edim == 2, "hopper: this hopper can only be apllied to edim = 2");
    for (int i(0); i < ndim; ++i) {
        misc::crasher::confirm(mass[i] == mass[0], "hopper: this hopper can work w/ identical mass on each dim");
    }

    // extract information
    vector<double> r(ndim);
    vector<double> p(ndim);
    vector< complex<double> > c(edim);
    int s;
    for (int i(0); i < ndim; ++i) {
        r[i] = state[i].real();
        p[i] = state[ndim + i].real();
    }
    for (int i(0); i < edim; ++i) {
        c[i] = state[ndim * 2 + i];
    }
    s = static_cast<int>(state[ndim * 2 + edim].real());

    const int from = s;
    const int to = 1 - s;

    // calc hop prob
    complex<double> vd = 0.0;
    for (int i(0); i < ndim; ++i) {
        vd += p[i] / mass[i] * dc[i][to+from*edim];
    }
    double g = -2 * dt * (c[from] * conj(c[to]) * vd).real() / (c[from] * conj(c[from])).real();
    double dE = eva[to] - eva[from];

    // random number
    if (randomer::rand() < g) {
        // momentum-rescaling direction: (x-direction)
        vector<double> n(ndim, 0.0);

        // Mehod #2
        const vector<double> dcR { dc[0][from+to*edim].real(), dc[1][from+to*edim].real()  };
        const vector<double> dcI { dc[0][from+to*edim].imag(), dc[1][from+to*edim].imag()  };
        const double diff_norm2 = norm2(dcR) - norm2(dcI);
        const double twice_eta0 = std::atan(-2 * (dcR[0] * dcI[0] + dcR[1] * dcI[1]) / diff_norm2);
        double eta;
        if (cos(twice_eta0) * diff_norm2 > 0.0) {
            eta = 0.5 * twice_eta0;
        }
        else {
            eta = 0.5 * twice_eta0 + 0.5 * M_PI;

        }
        // debug
        //eta = 0.0;
        const complex<double> eieta = exp(matrixop::IMAGIZ * eta);
        n[0] = (eieta * dc[0][from+to*edim]).real();
        n[1] = (eieta * dc[1][from+to*edim]).real();


        // hop
        if (norm(n) > 1e-40) {
            vector<double> pn = component(p, n);
            double pn_norm = norm(pn); 
            double tmp = pn_norm * pn_norm - 2 * mass[0] * dE; // use mass[0] for simplicity, masses along all dimension should be identical
            if (tmp > 0.0) {
                // hop accepted
                double pn_norm_new = sqrt(tmp);
                p += (pn_norm_new - pn_norm) / pn_norm * pn;

                // replace p & s
                for (int i(0); i < ndim; ++i) {
                    state[ndim + i].real(p[i]);
                }
                state[2 * ndim + edim] = to;
                return (from < to) ? HOP_UP : HOP_DN;
            }
            else {
                // hop frustrated
                return HOP_FR;
            }
        }

    }

    return HOP_RJ;
}


struct observer {

    public:
        int m_Nrec, m_irec;
        map< string, vector<double> > m_data_arr;
        const vector<string> m_keys { 
            "n0trans", "n0refl", "n1trans", "n1refl",
            "px0trans", "px0refl", "px1trans", "px1refl",
            "py0trans", "py0refl", "py1trans", "py1refl",
            "pz0trans", "pz0refl", "pz1trans", "pz1refl",
            "KE", "PE"
        };

    public:
        observer(const int Nrec) 
            : m_Nrec(Nrec), m_irec(0)
        {
            for (const string& key : m_keys) {
                m_data_arr.insert( std::make_pair(key, vector<double>(m_Nrec, 0.0)) );
            }
        }

        ~observer() = default;

    public:
        void add_record(const vector<state_t>& states, const vector<double>& mass) { 
            /*
             * record a row of data
             */

            // check 
            const int ndim = mass.size();
            const int edim = (states.at(0).size() - 1) / mass.size();

            misc::crasher::confirm(ndim == 3, "observer::add_record: ndim must be 3");
            misc::crasher::confirm(edim == 2, "observer::add_record: edim must be 2");
            misc::crasher::confirm(m_irec < m_Nrec, "observer::add_record: capacity reached");

            // population
            double n0trans = 0.0,  n0refl = 0.0, n1trans = 0.0, n1refl = 0.0;

            for_each(states.begin(), states.end(), 
                    [&n0trans, &n0refl, &n1trans, &n1refl,
                     &ndim, &edim, &mass] (const state_t& st) { 
                        int s = static_cast<int>(st[2 * ndim + edim].real());
                        double px = st[ndim].real();
                        if (s == 0) {
                            (px >= 0.0) ? n0trans += 1.0 : n0refl += 1.0;
                        }
                        else if (s == 1) {
                            (px >= 0.0) ? n1trans += 1.0 : n1refl += 1.0;
                        }
                    });

            m_data_arr["n0trans"].at(m_irec) += n0trans;
            m_data_arr["n0refl"].at(m_irec) += n0refl;
            m_data_arr["n1trans"].at(m_irec) += n1trans;
            m_data_arr["n1refl"].at(m_irec) += n1refl;

            // momentum
            double px0trans = 0.0, px0refl = 0.0, px1trans = 0.0, px1refl = 0.0;
            double py0trans = 0.0, py0refl = 0.0, py1trans = 0.0, py1refl = 0.0;
            double pz0trans = 0.0, pz0refl = 0.0, pz1trans = 0.0, pz1refl = 0.0;

            for_each(states.begin(), states.end(), 
                    [&px0trans, &px0refl, &px1trans, &px1refl, 
                     &py0trans, &py0refl, &py1trans, &py1refl,
                     &pz0trans, &pz0refl, &pz1trans, &pz1refl,
                     &ndim, &edim, &mass ] (const state_t& st) {
                        int s = static_cast<int>(st[2 * ndim + edim].real());
                        double px = st[ndim].real();
                        if (s == 0) {
                            if (px >= 0.0) {
                                px0trans += st[ndim].real();
                                py0trans += st[ndim + 1].real();
                                pz0trans += st[ndim + 2].real();
                            }
                            else {
                                px0refl += st[ndim].real();
                                py0refl += st[ndim + 1].real();
                                pz0refl += st[ndim + 2].real();
                            }
                        }
                        else if (s == 1){
                            if (px >= 0.0) {
                                px1trans += st[ndim].real();
                                py1trans += st[ndim + 1].real();
                                pz1trans += st[ndim + 2].real();
                            }
                            else {
                                px1refl += st[ndim].real();
                                py1refl += st[ndim + 1].real();
                                pz1refl += st[ndim + 2].real();
                            }
                        }
                    }
                    );

            m_data_arr["px0trans"].at(m_irec) += px0trans;
            m_data_arr["px0refl"].at(m_irec) += px0refl;
            m_data_arr["px1trans"].at(m_irec) += px1trans;
            m_data_arr["px1refl"].at(m_irec) += px1refl;
            m_data_arr["py0trans"].at(m_irec) += py0trans;
            m_data_arr["py0refl"].at(m_irec) += py0refl;
            m_data_arr["py1trans"].at(m_irec) += py1trans;
            m_data_arr["py1refl"].at(m_irec) += py1refl;
            m_data_arr["pz0trans"].at(m_irec) += pz0trans;
            m_data_arr["pz0refl"].at(m_irec) += pz0refl;
            m_data_arr["pz1trans"].at(m_irec) += pz1trans;
            m_data_arr["pz1refl"].at(m_irec) += pz1refl;

            // energy
            double KE = 0.0, PE = 0.0;
            for_each(states.begin(), states.end(), 
                    [&KE, &PE, &ndim, &edim, &mass] (const state_t& st) {
                        // extract info
                        vector<double> r(ndim);
                        vector<double> p(ndim);
                        for (int i(0); i < ndim; ++i) {
                            r[i] = st[i].real();
                            p[i] = st[ndim + i].real();
                        }
                        int s = static_cast<int>(st[2 * ndim + edim].real());

                        // KE
                        for (int i(0); i < ndim; ++i) {
                            KE += 0.5 / mass[i] * p[i] * p[i];
                        }

                        // PE 
                        vector<double> eva;
                        matrixop::hdiag(cal_H(r), eva);
                        PE += eva.at(s);
                    }
                    );
            m_data_arr["KE"].at(m_irec) = KE;
            m_data_arr["PE"].at(m_irec) = PE;

            // increase m_irec
            m_irec += 1;
        }

        void fill_unrecorded(const string& mode) { 
            /*
             * fill unrecorded data with the last record
             */
            if (m_irec > 0) {
                double val;
                for (const string& key : m_keys) {
                    if (mode == "zero") {
                        val = 0.0;
                    }
                    else if (mode == "last") {
                        val = m_data_arr[key][m_irec - 1];
                    }

                    fill(m_data_arr[key].begin() + m_irec, m_data_arr[key].end(), val);
                }
            }

            m_irec = m_Nrec;
        }

        void join_record() { 
            /*
             * collect all data to the master process
             */
            for (int r(1); r < MPIer::size; ++r) {
                if (MPIer::rank == r) {
                    for (const string& key : m_keys) {
                        MPIer::send(0, m_data_arr[key]);
                    }
                }
                else if (MPIer::master) {
                    vector<double> vbuf;
                    for (const string& key : m_keys) {
                        MPIer::recv(r, vbuf);
                        m_data_arr[key] += vbuf;
                    }
                }
                MPIer::barrier();
            }
            MPIer::barrier();
        }

        vector<string> get_keys() const {
            /*
             * get all keys
             */
            return m_keys;
        }

        double get_record(const string& key, int irec) const { 
            /*
             * get a piece of data 
             */
            auto it = m_data_arr.find(key);
            misc::crasher::confirm(it != m_data_arr.end(), "observer::get: the key does not exist.");
            return it->second.at(irec);
        }

        map<string, double> get_record(int irec) const { 
            /*
             * get a row of data 
             */
            map<string, double> row;
            for (const string& key : m_keys) {
                row.insert( make_pair(key, m_data_arr.at(key).at(irec)) );
            }
            return row;
        }
};


void fssh_nd_mpi() {
    /*
     * n-dimensional FSSH algorithm
     */

    // assign jobs
    const vector<int> my_jobs = MPIer::assign_job(Ntraj);
    const int my_Ntraj = my_jobs.size();


    // initialize trajectories
    vector<state_t> state(my_Ntraj);
    for (int itraj(0); itraj < my_Ntraj; ++itraj) {
        init_state(state[itraj], init_r, init_p, mass, init_s);
    }


    // hop statistics
    double hopup = 0.0, hopdn = 0.0, hopfr = 0.0, hoprj = 0.0;
    vector<double> hop_count(my_Ntraj, 0.0);
    vector<double> hop_count_summary(50, 0.0);

    // recorders
    int Nrec = Nstep / output_step;
    observer obs(Nrec);


    //  ----  MAIN LOOP  ---- //


    // last evt save
    vector< vector< complex<double> > > lastevt_save(my_Ntraj);

    for (int istep(0); istep < Nstep; ++istep) {
        for (int itraj(0); itraj < my_Ntraj; ++itraj) {
            if (check_end(state[itraj]) == false) {
                // assign last evt
                lastevt = move(lastevt_save[itraj]);
                // integrate t -> t + dt
                integrator(state[itraj], mass, istep * dt, dt);
                // hopper
                int hopflag = hopper(state[itraj], mass);
                switch (hopflag) {
                    case HOP_UP : { hopup += 1.0; hop_count[itraj] += 1.0; break; }
                    case HOP_DN : { hopdn += 1.0; hop_count[itraj] += 1.0; break; }
                    case HOP_FR : { hopfr += 1.0; break; }
                    case HOP_RJ : { hoprj += 1.0; break; }
                    default : break;
                }
                // save lastevt
                lastevt_save[itraj] = move(lastevt);
            }
        }

        if (istep % output_step == 0) {
            // data record
            obs.add_record(state, mass);

            // check end
            const bool end_flag = all_of(state.begin(), state.end(), check_end);
            if (end_flag == true) {
                // fill the rest
                obs.fill_unrecorded("last");
                break;
            }
        }
    }
    MPIer::barrier();


    // ----  COLLECT DATA  ---- //
    

    obs.join_record();
    MPIer::barrier();


    // ----  PROCESS & COLLECT HOP STATISTICS DATA  ---- //


    for_each(hop_count.begin(), hop_count.end(),
            [&hop_count_summary](double x) { hop_count_summary[static_cast<int>(x)] += 1.0; });
    MPIer::barrier();

    for (int r = 1; r < MPIer::size; ++r) {
        if (MPIer::rank == r) {
            MPIer::send(0, hopup, hopdn, hopfr, hoprj, hop_count_summary);
        }
        else if (MPIer::master) {
            double buf;
            vector<double> vbuf;

            MPIer::recv(r, buf); hopup += buf;
            MPIer::recv(r, buf); hopdn += buf;
            MPIer::recv(r, buf); hopfr += buf;
            MPIer::recv(r, buf); hoprj += buf;
            MPIer::recv(r, vbuf); hop_count_summary += vbuf;
        }
        MPIer::barrier();
    }
    MPIer::barrier();


    // ----  PROCESS & COLLECT TRAJ DATA  ---- //


    vector<double> xarr(my_Ntraj), yarr(my_Ntraj), zarr(my_Ntraj);
    vector<double> pxarr(my_Ntraj), pyarr(my_Ntraj), pzarr(my_Ntraj);
    vector<double> c0Rarr(my_Ntraj), c0Iarr(my_Ntraj);
    vector<double> c1Rarr(my_Ntraj), c1Iarr(my_Ntraj);
    vector<int> sarr(my_Ntraj);

    for (int itraj = 0; itraj < my_Ntraj; ++itraj) {
        xarr[itraj] = state[itraj][0].real();
        yarr[itraj] = state[itraj][1].real();
        zarr[itraj] = state[itraj][2].real();
        pxarr[itraj] = state[itraj][ndim+0].real();
        pyarr[itraj] = state[itraj][ndim+1].real();
        pzarr[itraj] = state[itraj][ndim+2].real();
        sarr[itraj] = static_cast<int>(state[itraj][2*ndim+edim].real());
        c0Rarr[itraj] = state[itraj][2*ndim+0].real();
        c0Iarr[itraj] = state[itraj][2*ndim+0].real();
        c1Rarr[itraj] = state[itraj][2*ndim+1].real();
        c1Iarr[itraj] = state[itraj][2*ndim+1].real();
    }

    for (int r = 1; r < MPIer::size; ++r) {
        if (MPIer::rank == r) {
            MPIer::send(0, xarr, yarr, zarr, pxarr, pyarr, pzarr,
                            sarr, c0Rarr, c0Iarr, c1Rarr, c1Iarr);
        }
        else if (MPIer::master) {
            vector<double> vbuf;
            vector<int> ibuf;

            MPIer::recv(r, vbuf); xarr.insert(xarr.end(), vbuf.begin(), vbuf.end());
            MPIer::recv(r, vbuf); yarr.insert(yarr.end(), vbuf.begin(), vbuf.end());
            MPIer::recv(r, vbuf); zarr.insert(zarr.end(), vbuf.begin(), vbuf.end());

            MPIer::recv(r, vbuf); pxarr.insert(pxarr.end(), vbuf.begin(), vbuf.end());
            MPIer::recv(r, vbuf); pyarr.insert(pyarr.end(), vbuf.begin(), vbuf.end());
            MPIer::recv(r, vbuf); pzarr.insert(pzarr.end(), vbuf.begin(), vbuf.end());

            MPIer::recv(r, ibuf); sarr.insert(sarr.end(), ibuf.begin(), ibuf.end());

            MPIer::recv(r, vbuf); c0Rarr.insert(c0Rarr.end(), vbuf.begin(), vbuf.end());
            MPIer::recv(r, vbuf); c0Iarr.insert(c0Iarr.end(), vbuf.begin(), vbuf.end());
            MPIer::recv(r, vbuf); c1Rarr.insert(c1Rarr.end(), vbuf.begin(), vbuf.end());
            MPIer::recv(r, vbuf); c1Iarr.insert(c1Iarr.end(), vbuf.begin(), vbuf.end());
        }
        MPIer::barrier();
    }
    MPIer::barrier();



    // ----  OUTPUT  ---- //


    if (MPIer::master) {
        // Output parameters
        output_potential_param();
        ioer::info("# FSSH parameters: ", 
                " Ntraj = ", Ntraj, " Nstep = ", Nstep, " dt = ", dt, 
                " mass = ", mass, 
                " init_r = ", init_r, " init_p = ", init_p, 
                " sigma_r = ", sigma_r, " sigma_p = ", sigma_p, 
                " init_s = ", init_s,
                " xwall_left = ", xwall_left, " xwall_right = ", xwall_right, 
                " output_step = ", output_step, " output_mod = ", output_mod
                );
        // Output header
        ioer::tabout(
                "#", "t", 
                "n0trans", "n0refl", "n1trans", "n1refl", 
                "px0trans", "py0trans", "px0refl", "py0refl", 
                "px1trans", "py1trans", "px1refl", "py1refl", 
                "Etot");
        // Output data
        map<string, double> row;
        for (int irec = 0; irec < Nrec; ++irec) {
            row = obs.get_record(irec);


            // average over all traj
            for (const string& key : obs.get_keys()) {
                row[key] /= Ntraj;
            }

            // partial average
            if (row["n0trans"] > 0.0) { row["px0trans"] /= row["n0trans"]; row["py0trans"] /= row["n0trans"]; }
            if (row["n0refl"]  > 0.0) { row["px0refl"]  /= row["n0refl"];  row["py0refl"]  /= row["n0refl"];  }
            if (row["n1trans"] > 0.0) { row["px1trans"] /= row["n1trans"]; row["py1trans"] /= row["n1trans"]; }
            if (row["n1refl"]  > 0.0) { row["px1refl"]  /= row["n1refl"];  row["py1refl"]  /= row["n1refl"];  }


            ioer::tabout(
                    "#", irec * output_step * dt, 
                    row["n0trans"], row["n0refl"], row["n1trans"], row["n1refl"],
                    row["px0trans"], row["py0trans"], row["px0refl"], row["py0refl"], 
                    row["px1trans"], row["py1trans"], row["px1refl"], row["py1refl"], 
                    row["KE"] + row["PE"]
                    );
        }

        // Output final results
        ioer::info("# final results: ");
        if (output_mod == "init_px") {
            ioer::tabout_nonewline(init_p[0]);
        }
        else if (output_mod == "init_s") {
            ioer::tabout_nonewline(init_s[1]);
        }

        ioer::tabout(
                row["n0trans"], row["n0refl"], row["n1trans"], row["n1refl"],
                row["px0trans"], row["py0trans"], row["px0refl"], row["py0refl"], 
                row["px1trans"], row["py1trans"], row["px1refl"], row["py1refl"], 
                row["KE"] + row["PE"]
                );


        // Output hop info
        ioer::info("# hop statistics: ");
        ioer::info("# hopup = ", hopup, " hopdn = ", hopdn, " hopfr = ", hopfr, " hopfr_rate = ", hopfr / (hopup + hopdn + hopfr));
        ioer::info("# hop count: ", hop_count_summary);

        // Output detailed trajectory info
        ioer::info("# traj info: ");
        ioer::tabout("# x", "y", "z", "px", "py", "pz", "s", "c0R", "c0I", "c1R", "c1I", "");
        for (int itraj = 0; itraj < Ntraj; ++itraj) {
            ioer::tabout(
                    xarr[itraj], yarr[itraj], zarr[itraj],
                    pxarr[itraj], pyarr[itraj], pzarr[itraj],
                    sarr[itraj],
                    c0Rarr[itraj], c0Iarr[itraj], 
                    c1Rarr[itraj], c1Iarr[itraj], 
                    ""
                    );
        }
    }
    MPIer::barrier();
}

int test() {
    vector<double> r(3, 0.0);
    for (double x(-8.0); x < 8.0; x += 0.1) {
        for (double y(-8.0); y < 8.0; y += 0.1) {
        r[0] = x;
        r[1] = y;

        cal_info_nume(r, eva, dc, F, lastevt);
        ioer::tabout(x, y, 
                eva[0], eva[1], 
                abs(dc[0][0+1*2]), abs(dc[1][0+1*2]), 
                F[0][0+0*2].real(), F[1][0+0*2].real(),
                F[0][1+1*2].real(), F[1][1+1*2].real()
                );
        }
    }
    // extract information
    /*
    vector<double> r(ndim, 0.0);
    vector<double> p(ndim, 0.0);
    vector<double> force(ndim, 0.0);
    int s = 1;

    ioer::tabout("# x", "y", "Fx", "Fy", "Fxmag", "Fymag", "Fxmag2", "Fymag2", "d10xR", "d10xI", "d10yR", "d10yI");

    for (double x = -5; x < 5; x += 0.01) {
        double y = -1.0;
        double px = 20.0;
        double py = 0.0;

        r[0] = x;
        r[1] = y;
        r[2] = 0.0;
        p[0] = px;
        p[1] = py;
        p[2] = 0.0;

        // calculte info 
        cal_info_nume(r, eva, dc, F, lastevt);
        for (int i(0); i < ndim; ++i) {
            force[i] = F[i][s + s * edim].real();
        }

        // cross(p, w) = F^mag
        const double w0 = 0.0, w1 = 0.0;
        double w2 = 0.0;
        for (int k(0); k < edim; ++k) {
            if (k != s) {
                w2 += (dc[0][s+k*edim] * dc[1][k+s*edim]).imag();
            }
        }
        w2 *= 2.0 / mass[2];

        vector<double> Fmag { w2 * p[1], -w2 * p[0], 0.0 };

        const complex<double> vdotdc = 
            p[0] * dc[0][0+s*edim] / mass[0] 
            + p[1] * dc[1][0+s*edim] / mass[1] 
            + p[2] * dc[2][0+s*edim] / mass[2]
            ; 

        vector<double> Fmag2 { 
            2.0 * (dc[0][s+0*edim] * vdotdc).imag(), 
            2.0 * (dc[1][s+0*edim] * vdotdc).imag(), 
            2.0 * (dc[2][s+0*edim] * vdotdc).imag()
        };

        ioer::tabout(x, y, force[0], force[1], Fmag[0], Fmag[1], Fmag2[0], Fmag2[1], 
                real(dc[0][s+0*edim]), 
                imag(dc[0][s+0*edim]), 
                real(dc[1][s+0*edim]), 
                imag(dc[1][s+0*edim]));
    }
    */

    return 0;
}

int main(int argc, char** argv) {
    //return(test());


    MPIer::setup();
    if (argc < 2) {
        if (MPIer::master) ioer::info("use --help for detailed info");
    }
    else {
        if (argparse(argc, argv) == false) {
            return 0;
        }
        randomer::seed(MPIer::assign_random_seed(seed));
        if (MPIer::master) timer::tic();
        fssh_nd_mpi();
        if (MPIer::master) ioer::info("# ", timer::toc());
    }
    MPIer::barrier();
    MPIer::finalize();
    return 0;
}
