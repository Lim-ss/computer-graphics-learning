//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>

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

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f> &normals)
{
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}

// Bresenham's line drawing algorithm
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end)
{
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;

    dx = x2 - x1;
    dy = y2 - y1;
    dx1 = fabs(dx);
    dy1 = fabs(dy);
    px = 2 * dy1 - dx1;
    py = 2 * dx1 - dy1;

    if (dy1 <= dx1)
    {
        if (dx >= 0)
        {
            x = x1;
            y = y1;
            xe = x2;
        }
        else
        {
            x = x2;
            y = y2;
            xe = x1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; x < xe; i++)
        {
            x = x + 1;
            if (px < 0)
            {
                px = px + 2 * dy1;
            }
            else
            {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0))
                {
                    y = y + 1;
                }
                else
                {
                    y = y - 1;
                }
                px = px + 2 * (dy1 - dx1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    }
    else
    {
        if (dy >= 0)
        {
            x = x1;
            y = y1;
            ye = y2;
        }
        else
        {
            x = x2;
            y = y2;
            ye = y1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; y < ye; i++)
        {
            y = y + 1;
            if (py <= 0)
            {
                py = py + 2 * dx1;
            }
            else
            {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0))
                {
                    x = x + 1;
                }
                else
                {
                    x = x - 1;
                }
                py = py + 2 * (dx1 - dy1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(int x, int y, const Vector4f *_v)
{
    Vector3f v[3];
    for (int i = 0; i < 3; i++)
        v[i] = {_v[i].x(), _v[i].y(), 1.0};
    Vector3f f0, f1, f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(x, y, 1.);
    if ((p.dot(f0) * f0.dot(v[2]) > 0) && (p.dot(f1) * f1.dot(v[0]) > 0) && (p.dot(f2) * f2.dot(v[1]) > 0))
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f *v)
{
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
    return {c1, c2, c3};
}

void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList)
{

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (const auto &t : TriangleList)
    {
        Triangle newtri = *t;

        std::array<Eigen::Vector4f, 3> mm{// 视图坐标
                                          (view * model * t->v[0]),
                                          (view * model * t->v[1]),
                                          (view * model * t->v[2])};

        std::array<Eigen::Vector3f, 3> viewspace_pos; // 三角形三个顶点在未做投影变换时在视图中的坐标，相当于相机到点的向量

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto &v)
                       {
                           return v.template head<3>(); // 这里的lambda函数返回值看不懂
                       });

        Eigen::Vector4f v[] = {// 屏幕坐标
                               mvp * t->v[0],
                               mvp * t->v[1],
                               mvp * t->v[2]};
        // Homogeneous division
        for (auto &vec : v)
        {
            vec.x() /= vec.w();
            vec.y() /= vec.w();
            vec.z() /= vec.w();
            // vec.w()=-vec.w();//使传入的深度值是正的，方便计算;其实没必要，因为求某个量的插值时分母和分子中都有w，符号约掉了
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose(); // 取逆再转置是要干嘛？
        Eigen::Vector4f n[] = {                                           // 视图坐标中，三个顶点的法向量，设成4f再传入3f给新三角是为了能够和变换矩阵做计算
                               inv_trans * to_vec4(t->normal[0], 0.0f),
                               inv_trans * to_vec4(t->normal[1], 0.0f),
                               inv_trans * to_vec4(t->normal[2], 0.0f)};

        // Viewport transformation
        for (auto &vert : v)
        {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 - f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            // screen space coordinates
            newtri.setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i)
        {
            // view space normal
            newtri.setNormal(i, n[i].head<3>());
        }

        newtri.setColor(0, 148, 121.0, 92.0);
        newtri.setColor(1, 148, 121.0, 92.0);
        newtri.setColor(2, 148, 121.0, 92.0);

        // Also pass view space vertice position这里指的是把视口空间坐标系中的z值通过顶点的第四个分量传进了三角形对象中
        rasterize_triangle(newtri, viewspace_pos); // newtri中的位置定义在屏幕坐标系，法线定义在视图坐标系，为了求颜色，函数参数也传入在视图坐标系的位置
    }
}

/*
以下插值函数也定义得十分抽象，像是给线性插值使用的，如果在rasterize_triangle()中用的话需要这样调用：
interpolate(alpha/v[0].w(), beta/v[1].w(), gamma/v[2].w(), t.color[0], t.color[1], t.color[2], 1 / wn);
第一个是我补的，不然这个函数没有能算float的重载
*/
static float interpolate(float alpha, float beta, float gamma, const float &vert1, const float &vert2, const float &vert3, float weight)
{
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f &vert1, const Eigen::Vector3f &vert2, const Eigen::Vector3f &vert3, float weight)
{
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f &vert1, const Eigen::Vector2f &vert2, const Eigen::Vector2f &vert3, float weight)
{
    auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
    auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

    u /= weight;
    v /= weight;

    return Eigen::Vector2f(u, v);
}

// Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle &t, const std::array<Eigen::Vector3f, 3> &view_pos)
{
    // TODO: From your HW3, get the triangle rasterization code.
    std::array<Eigen::Vector4f, 3ULL> v = {t.v[0], t.v[1], t.v[2]};
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
    for (float y = bottom; y <= top; y++)
    {
        for (float x = left; x <= right; x++)
        {
#ifndef SSAA_ON
            // TODO: Inside your rasterization loop:
            //    * v[i].w() is the vertex view space depth value z.
            //    * Z is interpolated view space depth for the current pixel
            //    * zp is depth between zNear and zFar, used for z-buffer
            auto ind = (height - 1 - y) * width + x; // ind为(x,y)点在（帧/深度）缓存中对应的下标

            // 作业2中重心坐标是在函数里计算的，因为结构化绑定返回三个值不是很方便，所以干脆写到外面来
            float alpha = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
            float beta = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
            float gamma = 1.0f - alpha - beta;
            float W = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            // TODO: Interpolate the attributes:
            // auto interpolated_color
            // auto interpolated_normal
            // auto interpolated_texcoords
            // auto interpolated_shadingcoords
            auto interpolated_z = interpolate(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w(), v[0].z(), v[1].z(), v[2].z(), 1 / W);
            auto interpolated_color = interpolate(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w(), t.color[0], t.color[1], t.color[2], 1 / W);
            auto interpolated_normal = interpolate(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w(), t.normal[0], t.normal[1], t.normal[2], 1 / W);
            auto interpolated_texcoords = interpolate(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w(), t.tex_coords[0], t.tex_coords[1], t.tex_coords[2], 1 / W);

            if (x < 0 || x >= width || y < 0 || y >= height) // 检测是否超过屏幕范围，防止越界访问缓存
                continue;
            if (interpolated_z > 0) // 如果三角形上对应该点在相机后面，则看不见，按理来说应该要判断是否在Znear和Zfar之间，但是由于rasterizer并未存该值，所以这里仅判断是否在相机前面（后面想的话随时可以加上这两个变量）
                continue;
            if (depth_buf[ind] < -interpolated_z) // 如果深度缓存比该点近，则三角形该点被遮住，负号是因为depth-buf里面存的是深度的绝对值，z为负数
                continue;

            // Use: fragment_shader_payload payload( interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
            // Use: payload.view_pos = interpolated_shadingcoords;
            // Use: Instead of passing the triangle's color directly to the frame buffer, pass the color to the shaders first to get the final color;
            // Use: auto pixel_color = fragment_shader(payload);
            if (insideTriangle(x, y, t.v))
            {
                depth_buf[ind] = -interpolated_z; // SSAA时，不能放在前面，否则不在三角形内的点的深度缓存也会错误地被更新，为了统一，这里也放if里面
                Eigen::Vector2i point;
                point << x, y;

                Vector3f temp_color = {255*255,255*255,255*255};
                set_pixel(point, interpolated_color*255);
            }
#endif
        }
    }
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
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);

    texture = std::nullopt;
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height - y) * width + x;
}

void rst::rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color)
{
    // old index: auto ind = point.y() + point.x() * width;
    int ind = (height - point.y()) * width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}
