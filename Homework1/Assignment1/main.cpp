#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

constexpr double MY_PI = 3.1415926;
// 视图变换，没有进行旋转，只进行了平移，相当于默认原本的方向就是正确的
Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1, -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}
// 物体变换
Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model; //= Eigen::Matrix4f::Identity();

    model << cos(rotation_angle * MY_PI / 180), -sin(rotation_angle * MY_PI / 180), 0, 0,
        sin(rotation_angle * MY_PI / 180), cos(rotation_angle * MY_PI / 180), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.

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
    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.

    return projection;
}

// 获得一个绕某顶点某向量旋转某角度的变换矩阵
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
int main(int argc, const char **argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3)
    {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4)
        {
            filename = std::string(argv[3]);
        }
    }

    rst::rasterizer r(700, 700); // 创建一个rasterizer对象r，初始化其width和height

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}, {0, 1, -1}, {1, 0, -2}, {2, 2, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}, {3, 4, 5}};

    auto pos_id = r.load_positions(pos); // 将pos传入r中，存放在键值对pos_buf中，并返回存放该键的结构体
    auto ind_id = r.load_indices(ind);   // 同上，存放在键值对ind_buf中

    int key = 0;
    int frame_count = 0;
    Eigen::Vector3f rotate_axis; // 旋转轴
    Eigen::Vector3f vertices;    // 旋转顶点
    rotate_axis << 1, 1, 0;
    vertices << 0, 0, -2;
    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth); // 清空frame_buf和depth_buf，分别对应颜色和深度，由于函数内部用位运算判断该清空哪个buffer，因此可以通过位或一次性清空两者

        // r.set_model(get_model_matrix(angle));
        r.set_model(get_rotation(vertices, rotate_axis, angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, -0.1, -50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data()); // 利用r中原有的frame_buf创建的一个700*700的32位浮点数三通道图像存储器（数据不拷贝）
        image.convertTo(image, CV_8UC3, 1.0f);

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

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        // std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a')
        {
            angle += 10;
        }
        else if (key == 'd')
        {
            angle -= 10;
        }
    }

    return 0;
}
