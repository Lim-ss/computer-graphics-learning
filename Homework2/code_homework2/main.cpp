//// clang-format off
#include <iostream>
#include <opencv2/opencv.hpp>
#include "rasterizer.hpp"
#include "global.hpp"
#include "Triangle.hpp"


constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0],
        0, 1, 0, -eye_pos[1],
        0, 0, 1, -eye_pos[2],
        0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model;

    model << cos(rotation_angle * MY_PI / 180), -sin(rotation_angle * MY_PI / 180), 0, 0,
        sin(rotation_angle * MY_PI / 180), cos(rotation_angle * MY_PI / 180), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.

    return model;
}

// 获得一个绕某顶点某向量旋转某角度的变换矩阵,get_model_matrix()的升级版
Eigen::Matrix4f get_rotation(Vector3f vertices, Vector3f axis, float angle)
{
    Eigen::Vector3f n = axis.normalized(); //= Eigen::Matrix4f::Identity();
    Eigen::Matrix4f translate_1;           // 平移到原点的变换矩阵
    Eigen::Matrix4f translate_2;           // 平移回原位置的变换矩阵
    Eigen::Matrix4f rotate;
    Eigen::Matrix3f N;
    Eigen::Matrix4f model;
    N << 0, -n.z(), n.y(),
        n.z(), 0, -n.x(),
        -n.y(), n.x(), 0;
    N = sin(angle * MY_PI / 180) * N;
    N = N + cos(angle * MY_PI / 180) * Eigen::Matrix3f::Identity() + (1 - cos(angle * MY_PI / 180)) * n * n.transpose();
    rotate << N.row(0), 0,
        N.row(1), 0,
        N.row(2), 0,
        0, 0, 0, 1;
    translate_1 << 1, 0, 0, -vertices.x(),
        0, 1, 0, -vertices.y(),
        0, 0, 1, -vertices.z(),
        0, 0, 0, 1;
    translate_2 << 1, 0, 0, vertices.x(),
        0, 1, 0, vertices.y(),
        0, 0, 1, vertices.z(),
        0, 0, 0, 1;
    model = translate_2 * rotate * translate_1;
    // std::cout<<"当前model矩阵为:"<<std::endl<<model<<std::endl;//debug
    return model;
}
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function
    Eigen::Matrix4f projection;
    Eigen::Matrix4f orthographic, orthographic_scale, orthographic_translate;
    Eigen::Matrix4f projection_to_orthographic;

    float top = -zNear * tan(0.5 * eye_fov * MY_PI / 180);
    float bottom = -top;
    float left = -top * aspect_ratio;
    float right = -left;

    orthographic_scale << 2 / (right - left), 0, 0, 0,
        0, 2 / (top - bottom), 0, 0,
        0, 0, 2 / (zNear - zFar), 0,
        0, 0, 0, 1;
    orthographic_translate << 1, 0, 0, -(right + left) / 2,
        0, 1, 0, -(top + bottom) / 2,
        0, 0, 1, -(zNear + zFar) / 2,
        0, 0, 0, 1;
    projection_to_orthographic << zNear, 0, 0, 0, // ppt上的n和f都是绝对值,但该矩阵算出来和ppt长一样
        0, zNear, 0, 0,
        0, 0, zNear + zFar, -zNear * zFar,
        0, 0, 1, 0;
    orthographic = orthographic_scale * orthographic_translate;
    projection = orthographic * projection_to_orthographic;

    return projection;
}

int main(int argc, const char **argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc == 2)
    {
        command_line = true;
        filename = std::string(argv[1]);
    }

    rst::rasterizer r(PIXEL_X, PIXEL_Y);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    // 给get_rotation()——————get_model_matrix()的升级版用的
    Eigen::Vector3f rotate_axis; // 旋转轴
    Eigen::Vector3f vertices;    // 旋转顶点
    rotate_axis << 1, 0, 0;
    vertices << 0, 0, -3.5;

    std::vector<Eigen::Vector3f> pos{
        {2, 0, -2},
        {0, 2, -2},
        {-2, 0, -2},
        {3.5, -1, -5},
        {2.5, 1.5, -5},
        {-1, 0.5, -5}};

    std::vector<Eigen::Vector3i> ind{
        {0, 1, 2},
        {3, 4, 5}};

    std::vector<Eigen::Vector3f> cols{
        {217.0, 238.0, 185.0},
        {217.0, 238.0, 185.0},
        {217.0, 238.0, 185.0},
        {185.0, 217.0, 238.0},
        {185.0, 217.0, 238.0},
        {185.0, 217.0, 238.0}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);
    auto col_id = r.load_colors(cols);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        // r.set_model(get_model_matrix(angle));
        r.set_model(get_rotation(vertices, rotate_axis, angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, -0.1, -50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
    #ifndef DEPTH_PICTURE
        cv::Mat image(PIXEL_X, PIXEL_Y, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    #endif
    #ifdef DEPTH_PICTURE
        cv::Mat image(PIXEL_X, PIXEL_Y, CV_32FC1, r.depth_buffer().data());
        image.convertTo(image, CV_8UC1, 1.0f);
    #endif
        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27)
    {
        
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        // r.set_model(get_model_matrix(angle));
        r.set_model(get_rotation(vertices, rotate_axis, angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, -0.1, -50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        #ifndef DEPTH_PICTURE
        cv::Mat image(PIXEL_X, PIXEL_Y, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imshow("image", image);
        #endif
        #ifdef DEPTH_PICTURE
        cv::Mat image(PIXEL_X, PIXEL_Y, CV_32FC1, r.depth_buffer().data());
        image.convertTo(image, CV_8UC1, 1.0f);
        cv::imshow("image", image);
        #endif
        //std::cout << "frame count: " << frame_count++ << '\n';
        key = cv::waitKey(10);
        if (key == 'a')
        {
            angle += 10;
        }
        else if (key == 'd')
        {
            angle -= 10;
        }
        // else
        // {
        //     continue;
        // }
    }
    return 0;
}
// clang-format on