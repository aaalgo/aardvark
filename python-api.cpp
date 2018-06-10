#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
using namespace boost::python;
namespace np = boost::python::numpy;

namespace {
    using std::istringstream;
    using std::ostringstream;
    using std::string;
    using std::runtime_error;
    using std::cerr;
    using std::endl;
    using std::vector;

    float box_area (float const *b) {
        return (b[2] - b[0]) * (b[3] - b[1]);
    }

#if 0
    float box_iarea (float const *b1, float const *b2) {
        float ibox[] = {std::max(b1[0], b2[0]),
                        std::max(b1[1], b2[1]),
                        std::min(b1[2], b2[2]),
                        std::min(b1[3], b2[3])};
        std::cerr << "AAA " << ibox[0] << " " << ibox[1] << " " << ibox[2] << " " << ibox[3] << std::endl;
        return box_area(ibox);
    }
#endif

    float iou_score (float const *b1, float const *b2) {
        float ibox[] = {std::max(b1[0], b2[0]),
                        std::max(b1[1], b2[1]),
                        std::min(b1[2], b2[2]),
                        std::min(b1[3], b2[3])};
        if (ibox[0] >= ibox[2]) return 0;
        if (ibox[1] >= ibox[3]) return 0;
        float ia = box_area(ibox);
        float ua = box_area(b1) + box_area(b2) - ia;
        return ia / (ua + 1.0);
    }

    class GTMatcher {
        float iou_th;
        int max;
    public:
        GTMatcher (float th_, int max_): iou_th(th_), max(max_) {
        }

        list apply (np::ndarray boxes,
                    np::ndarray box_ind_,
                    np::ndarray gt_boxes) {
            vector<std::pair<int, int>> match;

            CHECK(boxes.get_nd() == 2);
            CHECK(boxes.shape(0) == 0 || boxes.shape(1) == 4);
            CHECK(gt_boxes.get_nd() == 2);
            CHECK(gt_boxes.shape(0) == 0 || gt_boxes.shape(1) >= 7);
            // assign prediction to gt_boxes
            // algorithm:
            //      for each gt box pick the best match
            int nb = boxes.shape(0);
            int ng = gt_boxes.shape(0);
            CHECK(nb == box_ind_.shape(0));
            vector<bool> used(nb, false);

            int32_t const *box_ind = (int32_t const *)(box_ind_.get_data());
            for (int i = 0; i < ng; ++i) {
                // i-th gt box
                float const *gt = (float const *)(gt_boxes.get_data() + gt_boxes.strides(0) * i);
                int ind = gt[0];
                gt = gt + 3;    // the box parameters
                float iou = iou_th;
                int best = -1;
                for (int j = 0; j < nb; ++j) {
                    if (box_ind[j] != ind) continue; // not the same image
                    if (used[j]) continue;
                    float const *b = (float const *)(boxes.get_data() + boxes.strides(0) * j);
                    float s = iou_score(gt, b);
                    if (s > iou) {
                        iou = s;
                        best = j;
                    }
                }
                if (best >= 0) {
                    /*
                    float const *b = (float const *)(boxes.get_data() + boxes.strides(0) * best);
                    std::cerr << "YYY " << iou << ' ' << b[0] << ' ' << b[1] << ' ' << b[2] << ' ' << b[3] << std::endl;
                    float ia = box_iarea(gt, b);
                    std::cerr << iou_score(gt, b) << "  " << ia << "    " << box_area(gt) << "    " << box_area(b) << std::endl;
                    */
                    match.emplace_back(best, i);
                    used[best] = true;
                }
            }

            list r;
            np::ndarray cnt = np::zeros(make_tuple(), np::dtype::get_builtin<float>());
            *(float *)cnt.get_data() = match.size();

            if (match.size() > max) {
                std::random_shuffle(match.begin(), match.end());
                match.resize(max);
            }

            np::ndarray idx1 = np::zeros(make_tuple(match.size()), np::dtype::get_builtin<int32_t>());
            np::ndarray idx2 = np::zeros(make_tuple(match.size()), np::dtype::get_builtin<int32_t>());
            
            //np::ndarray cnt = np::zeros(mask_tuple(

            int32_t *p1 = (int32_t *)idx1.get_data();
            int32_t *p2 = (int32_t *)idx2.get_data();
            for (auto const &p: match) {
                *p1 = p.first;
                *p2 = p.second;
                ++p1;
                ++p2;
            }
            r.append(cnt);
            r.append(idx1);
            r.append(idx2);
            return r;
        }
    };

    class MaskExtractor {
        cv::Size sz;
    public:
        MaskExtractor (int width, int height): sz(width, height) {
        }

        np::ndarray apply (np::ndarray images,
                    np::ndarray gt_boxes,
                    np::ndarray boxes) {
            int n = 0;
            int H = images.shape(1);
            int W = images.shape(2);
            int C = images.shape(3);
            CHECK(C == 1);

            do {
                CHECK(images.get_nd() == 4);
                CHECK(gt_boxes.get_nd() == 2);
                CHECK(boxes.get_nd() == 2);
                CHECK(gt_boxes.shape(0) == boxes.shape(0));
                if (gt_boxes.shape(0) == 0) break;
                CHECK(gt_boxes.shape(1) >= 3);
                if (boxes.shape(0) == 0) break;
                CHECK(boxes.shape(1) == 4);
                n = gt_boxes.shape(0);
            } while(false);

            np::ndarray masks = np::zeros(make_tuple(n, sz.height, sz.width, 1), np::dtype::get_builtin<float>());

#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                float *gt_box = (float *)(gt_boxes.get_data() + i * gt_boxes.strides(0));
                float *box = (float *)(boxes.get_data() + i * boxes.strides(0));
                int index(gt_box[0]);
                int tag(gt_box[2]);
                cv::Mat image(H, W, CV_32F, images.get_data() + index * images.strides(0));
                float *mask_begin =  (float *)(masks.get_data() + i * masks.strides(0));
                cv::Mat mask(sz, CV_32F, mask_begin);

                int x1 = int(round(box[0]));
                int y1 = int(round(box[1]));
                int x2 = int(round(box[2]));
                int y2 = int(round(box[3]));
                CHECK(x1 >= 0);
                CHECK(y1 >= 0);
                CHECK(x2 < W);
                CHECK(y2 < H);
                cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
                cv::Mat from = image(roi).clone();
                for (float *p = from.ptr<float>(0); p < from.ptr<float>(from.rows); ++p) {
                    if (p[0] == tag) { p[0] = 1.0;}
                    else p[0] = 0.0;
                }
                cv::resize(from, mask, sz, 0, 0);
            }
            return masks;
        }
    };
}

BOOST_PYTHON_MODULE(cpp)
{
    np::initialize();
    class_<GTMatcher>("GTMatcher", init<float, int>())
        .def("apply", &GTMatcher::apply)
    ;
    class_<MaskExtractor>("MaskExtractor", init<int, int>())
        .def("apply", &MaskExtractor::apply)
    ;
}

