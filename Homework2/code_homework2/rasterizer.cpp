////clang-format off
// 上一行两斜线的时候临时禁用自动格式化
//  Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "global.hpp"

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}
// 这个函数为什么要给参数直接给定一个定值，那还传啥参数？？另外，这种与光栅器类本身没什么关系的函数真的适合定义在类中吗，就算定义，是不是改为静态函数更好
auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

// 判断点是否在三角形中，改为静态函数是否更好呢
static bool insideTriangle(float x, float y, const Vector3f *_v)
{
    Eigen::Vector3f a, b;
    Eigen::Vector3f inner_product_1, inner_product_2, inner_product_3; // 反正只用判断正负，不必用float
    a << x - _v[0].x(), y - _v[0].y(), 0;
    b << _v[1].x() - _v[0].x(), _v[1].y() - _v[0].y(), 0;
    inner_product_1 = a.cross(b);
    a << x - _v[1].x(), y - _v[1].y(), 0;
    b << _v[2].x() - _v[1].x(), _v[2].y() - _v[1].y(), 0;
    inner_product_2 = a.cross(b);
    a << x - _v[2].x(), y - _v[2].y(), 0;
    b << _v[0].x() - _v[2].x(), _v[0].y() - _v[2].y(), 0;
    inner_product_3 = a.cross(b);
    if ((inner_product_1.z() > 0 && inner_product_2.z() > 0 && inner_product_3.z() > 0) || (inner_product_1.z() < 0 && inner_product_2.z() < 0 && inner_product_3.z() < 0))
        return true;
    else
        return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f *v)
{
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
    return {c1, c2, c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto &buf = pos_buf[pos_buffer.pos_id];
    auto &ind = ind_buf[ind_buffer.ind_id];
    auto &col = col_buf[col_buffer.col_id];
    /*
    这里写死了，很不好，是不是应该把(eye_fov,aspect_ratio,zNear,zFar)，(eye_pos),(rotation_vertices, rotate_axis, angle)等
    放进光栅器类里，然后变换矩阵在内部计算，而不是从外部传入变换矩阵呢，特别是前两组，毕竟投影变换和视口变换不属于物体本身性质的改变，只和光栅化的过程有关
    另外，既然后面光栅化用到的坐标都是在长方体里的，当初为什么要变换到[-1~1]^3中，然后用的时候又变回来呢？
    */
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;
    Eigen::Matrix4f mvp = projection * view * model;
    for (auto &i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
            mvp * to_vec4(buf[i[0]], 1.0f),
            mvp * to_vec4(buf[i[1]], 1.0f),
            mvp * to_vec4(buf[i[2]], 1.0f)};
        // Homogeneous division
        for (auto &vec : v)
        {
            vec /= vec.w();
        }
        // Viewport transformation
        for (auto &vert : v)
        {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 - f2;
        }
        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);
        rasterize_triangle(t);
    }
}

// Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle &t)
{
    auto v = t.toVector4();
    // TODO : Find out the bounding box of current triangle.
    // std::max_element的第三个参数是lambda表达式，用于规定比较方式
    int top = std::max_element(v.begin(), v.end(), [](const Eigen::Vector4f &v1, const Eigen::Vector4f &v2)
                               { return (v1.y() < v2.y()); })
                  ->y(); // 此处发生了隐式类型转换
    int bottom = std::min_element(v.begin(), v.end(), [](const Eigen::Vector4f &v1, const Eigen::Vector4f &v2)
                                  { return (v1.y() < v2.y()); })
                     ->y();
    int right = std::max_element(v.begin(), v.end(), [](const Eigen::Vector4f &v1, const Eigen::Vector4f &v2)
                                 { return (v1.x() < v2.x()); })
                    ->x();
    int left = std::min_element(v.begin(), v.end(), [](const Eigen::Vector4f &v1, const Eigen::Vector4f &v2)
                                { return (v1.x() < v2.x()); })
                   ->x();
    std::cout << "三角形包围盒: top=" << top << "bottom=" << bottom << "right=" << right << "left=" << left << std::endl;
    //  iterate through the pixel and find if the current pixel is inside the triangle
    for (float y = bottom; y <= top; y++)
    {
        for (float x = left; x <= right; x++)
        {
#ifndef SSAA_ON
            auto ind = (height - 1 - y) * width + x; // ind为(x,y)点在（帧/深度）缓存中对应的下标
            /*
            以下三行用插值法求该像素点(x，y)对应三角形上的z
            下面的做法是有问题的，具体错误参考https://zhuanlan.zhihu.com/p/509902950
            关于透视矫正插值参考https://zhuanlan.zhihu.com/p/403259571
            简单理解就是，矫正插值是需要知道三角形三个点在视口坐标系（也就是投影变换进行之前）的深度的，而这个深度一般保存在v的第四个数中传进来
            （注意次数v不能算是一个齐次坐标，应该看成前三个值是屏幕空间中的坐标，第四个值是视口空间中的z坐标）
            然而，三角形类中顶点是三维的，所以在draw函数中，直接抹掉了第四个分量，传进来之后又补回个1，导致插值退化为线性插值
            这在homeworks3中得以改善，看到那里的三角形类已经是四维的了，可以直接通过第四个分类传入视口空间中的深度，且齐次除法时没有把w也一起除成1
            */
            auto [alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
            float wn = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            float z_interpolated = wn * (alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w());

            if (x < 0 || x >= width || y < 0 || y >= height) // 检测是否超过屏幕范围，防止越界访问缓存
                continue;
            if (z_interpolated > 0) // 如果三角形上对应该点在相机后面，则看不见，按理来说应该要判断是否在Znear和Zfar之间，但是由于rasterizer并未存该值，所以这里仅判断是否在相机前面（后面想的话随时可以加上这两个变量）
                continue;
            if (depth_buf1[ind] < -z_interpolated) // 如果深度缓存比该点近，则三角形该点被遮住，负号是因为depth-buf里面存的是深度的绝对值，z为负数
                continue;
            if (insideTriangle(x, y, t.v))
            {
                depth_buf1[ind] = -z_interpolated; // SSAA时，不能放在前面，否则不在三角形内的点的深度缓存也会错误地被更新，为了统一，这里也放if里面
                Eigen::Vector3f point;
                point << x, y, z_interpolated;
                set_pixel(point, t.getColor());
            }
#endif
#ifdef SSAA_ON
            if (x < 0 || x >= width || y < 0 || y >= height)
                continue;

            auto ind = (height - 1 - y) * width + x;
            auto [alpha1, beta1, gamma1] = computeBarycentric2D(x + 0.25, y + 0.25, t.v);
            float wn = 1.0 / (alpha1 / v[0].w() + beta1 / v[1].w() + gamma1 / v[2].w());
            float z_interpolated = alpha1 * v[0].z() / v[0].w() + beta1 * v[1].z() / v[1].w() + gamma1 * v[2].z() / v[2].w();
            z_interpolated *= wn;
            if (z_interpolated < 0 && depth_buf1[ind] > -z_interpolated && insideTriangle(x + 0.25, y + 0.25, t.v))
            {
                depth_buf1[ind] = -z_interpolated;
                frame_buf[ind] += 0.25 * t.getColor();
            }

            auto [alpha2, beta2, gamma2] = computeBarycentric2D(x + 0.25, y - 0.25, t.v);
            wn = 1.0 / (alpha2 / v[0].w() + beta2 / v[1].w() + gamma2 / v[2].w());
            z_interpolated = alpha2 * v[0].z() / v[0].w() + beta2 * v[1].z() / v[1].w() + gamma2 * v[2].z() / v[2].w();
            z_interpolated *= wn;
            if (z_interpolated < 0 && depth_buf2[ind] > -z_interpolated && insideTriangle(x + 0.25, y - 0.25, t.v))
            {
                depth_buf2[ind] = -z_interpolated;
                frame_buf[ind] += 0.25 * t.getColor();
            }

            auto [alpha3, beta3, gamma3] = computeBarycentric2D(x - 0.25, y + 0.25, t.v);
            wn = 1.0 / (alpha3 / v[0].w() + beta3 / v[1].w() + gamma3 / v[2].w());
            z_interpolated = alpha3 * v[0].z() / v[0].w() + beta3 * v[1].z() / v[1].w() + gamma3 * v[2].z() / v[2].w();
            z_interpolated *= wn;
            if (z_interpolated < 0 && depth_buf3[ind] > -z_interpolated && insideTriangle(x - 0.25, y + 0.25, t.v))
            {
                depth_buf3[ind] = -z_interpolated;
                frame_buf[ind] += 0.25 * t.getColor();
            }

            auto [alpha4, beta4, gamma4] = computeBarycentric2D(x - 0.25, y - 0.25, t.v);
            wn = 1.0 / (alpha4 / v[0].w() + beta4 / v[1].w() + gamma4 / v[2].w());
            z_interpolated = alpha4 * v[0].z() / v[0].w() + beta4 * v[1].z() / v[1].w() + gamma4 * v[2].z() / v[2].w();
            z_interpolated *= wn;
            if (z_interpolated < 0 && depth_buf4[ind] > -z_interpolated && insideTriangle(x - 0.25, y - 0.25, t.v))
            {
                depth_buf4[ind] = -z_interpolated;
                frame_buf[ind] += 0.25 * t.getColor();
            }
#endif
        }
    }
    //

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
}

void rst::rasterizer::set_model(const Eigen::Matrix4f &m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f &v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf1.begin(), depth_buf1.end(), std::numeric_limits<float>::infinity());
#ifdef SSAA_ON
        std::fill(depth_buf2.begin(), depth_buf2.end(), std::numeric_limits<float>::infinity());
        std::fill(depth_buf3.begin(), depth_buf3.end(), std::numeric_limits<float>::infinity());
        std::fill(depth_buf4.begin(), depth_buf4.end(), std::numeric_limits<float>::infinity());
#endif
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf1.resize(w * h);
#ifdef SSAA_ON
    depth_buf2.resize(w * h);
    depth_buf3.resize(w * h);
    depth_buf4.resize(w * h);
#endif
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height - 1 - y) * width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f &point, const Eigen::Vector3f &color)
{
    // old index: auto ind = point.y() + point.x() * width;

    auto ind = (height - 1 - point.y()) * width + point.x(); // ind为(x,y)点在（帧/深度）缓存中对应的下标
    frame_buf[ind] = color;

    return;
}

//clang-format on