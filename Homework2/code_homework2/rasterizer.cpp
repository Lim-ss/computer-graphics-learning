////clang-format off
// ��һ����б�ߵ�ʱ����ʱ�����Զ���ʽ��
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
// �������ΪʲôҪ������ֱ�Ӹ���һ����ֵ���ǻ���ɶ�����������⣬�������դ���౾��ûʲô��ϵ�ĺ�������ʺ϶����������𣬾��㶨�壬�ǲ��Ǹ�Ϊ��̬��������
auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

// �жϵ��Ƿ����������У���Ϊ��̬�����Ƿ������
static bool insideTriangle(float x, float y, const Vector3f *_v)
{
    Eigen::Vector3f a, b;
    Eigen::Vector3f inner_product_1, inner_product_2, inner_product_3; // ����ֻ���ж�������������float
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
    ����д���ˣ��ܲ��ã��ǲ���Ӧ�ð�(eye_fov,aspect_ratio,zNear,zFar)��(eye_pos),(rotation_vertices, rotate_axis, angle)��
    �Ž���դ�����Ȼ��任�������ڲ����㣬�����Ǵ��ⲿ����任�����أ��ر���ǰ���飬�Ͼ�ͶӰ�任���ӿڱ任���������屾�����ʵĸı䣬ֻ�͹�դ���Ĺ����й�
    ���⣬��Ȼ�����դ���õ������궼���ڳ�������ģ�����ΪʲôҪ�任��[-1~1]^3�У�Ȼ���õ�ʱ���ֱ�����أ�
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
    // std::max_element�ĵ�����������lambda���ʽ�����ڹ涨�ȽϷ�ʽ
    int top = std::max_element(v.begin(), v.end(), [](const Eigen::Vector4f &v1, const Eigen::Vector4f &v2)
                               { return (v1.y() < v2.y()); })
                  ->y(); // �˴���������ʽ����ת��
    int bottom = std::min_element(v.begin(), v.end(), [](const Eigen::Vector4f &v1, const Eigen::Vector4f &v2)
                                  { return (v1.y() < v2.y()); })
                     ->y();
    int right = std::max_element(v.begin(), v.end(), [](const Eigen::Vector4f &v1, const Eigen::Vector4f &v2)
                                 { return (v1.x() < v2.x()); })
                    ->x();
    int left = std::min_element(v.begin(), v.end(), [](const Eigen::Vector4f &v1, const Eigen::Vector4f &v2)
                                { return (v1.x() < v2.x()); })
                   ->x();
    std::cout << "�����ΰ�Χ��: top=" << top << "bottom=" << bottom << "right=" << right << "left=" << left << std::endl;
    //  iterate through the pixel and find if the current pixel is inside the triangle
    for (float y = bottom; y <= top; y++)
    {
        for (float x = left; x <= right; x++)
        {
#ifndef SSAA_ON
            auto ind = (height - 1 - y) * width + x; // indΪ(x,y)���ڣ�֡/��ȣ������ж�Ӧ���±�
            /*
            ���������ò�ֵ��������ص�(x��y)��Ӧ�������ϵ�z
            �����������������ģ��������ο�https://zhuanlan.zhihu.com/p/509902950
            ����͸�ӽ�����ֵ�ο�https://zhuanlan.zhihu.com/p/403259571
            �������ǣ�������ֵ����Ҫ֪�����������������ӿ�����ϵ��Ҳ����ͶӰ�任����֮ǰ������ȵģ���������һ�㱣����v�ĵ��ĸ����д�����
            ��ע�����v��������һ��������꣬Ӧ�ÿ���ǰ����ֵ����Ļ�ռ��е����꣬���ĸ�ֵ���ӿڿռ��е�z���꣩
            Ȼ�������������ж�������ά�ģ�������draw�����У�ֱ��Ĩ���˵��ĸ�������������֮���ֲ��ظ�1�����²�ֵ�˻�Ϊ���Բ�ֵ
            ����homeworks3�е��Ը��ƣ�������������������Ѿ�����ά���ˣ�����ֱ��ͨ�����ĸ����ഫ���ӿڿռ��е���ȣ�����γ���ʱû�а�wҲһ�����1
            */
            auto [alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
            float wn = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            float z_interpolated = wn * (alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w());

            if (x < 0 || x >= width || y < 0 || y >= height) // ����Ƿ񳬹���Ļ��Χ����ֹԽ����ʻ���
                continue;
            if (z_interpolated > 0) // ����������϶�Ӧ�õ���������棬�򿴲�����������˵Ӧ��Ҫ�ж��Ƿ���Znear��Zfar֮�䣬��������rasterizer��δ���ֵ������������ж��Ƿ������ǰ�棨������Ļ���ʱ���Լ���������������
                continue;
            if (depth_buf1[ind] < -z_interpolated) // �����Ȼ���ȸõ�����������θõ㱻��ס����������Ϊdepth-buf����������ȵľ���ֵ��zΪ����
                continue;
            if (insideTriangle(x, y, t.v))
            {
                depth_buf1[ind] = -z_interpolated; // SSAAʱ�����ܷ���ǰ�棬�������������ڵĵ����Ȼ���Ҳ�����ر����£�Ϊ��ͳһ������Ҳ��if����
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

    auto ind = (height - 1 - point.y()) * width + point.x(); // indΪ(x,y)���ڣ�֡/��ȣ������ж�Ӧ���±�
    frame_buf[ind] = color;

    return;
}

//clang-format on