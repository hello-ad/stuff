#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/split.hpp>
#include <vector>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>

using namespace std;
using namespace cv;
using namespace caffe;

typedef pair<string, float> Prediction;

static bool PairCompare(const std::pair<float, int>& lhs,
			const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}
static std::vector<int> Argmax(const std::vector<float>& v, int N)     {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
	
	std::vector<int> result;
	for (int i = 0; i < N; ++i)
	  result.push_back(pairs[i].second);
	return result;
}

class classifier {
	public:
		classifier(string model_file, 
			string trained_file,
			bool set_mean,
			bool set_regular,
			string mode,
			int cont);

		int set_cont(int cont);
		
		vector<float> calc(vector<Mat> imgs);
		vector<float> calc(vector<float> x, vector<float> y);
		bool check_blob(string blob_name);
		bool check_layer(string layer_name);
		const boost::shared_ptr<Layer<float> >get_layer(string name);
		const boost::shared_ptr<Blob<float> >get_blob(string name);
	private:
		shared_ptr<Net<float> > net_;
		cv::Size input_geometry_;
		int num_channels_;
		bool set_mean_;
		bool set_regular_;
		cv::Mat mean_;
		string mode_;
		int cont_;
};

const boost::shared_ptr<Blob<float> > classifier::get_blob(string name) {
	return net_->blob_by_name(name);
}
const boost::shared_ptr<Layer<float> > classifier::get_layer(string name) {
	return net_->layer_by_name(name);
}
bool classifier::check_layer(string layer_name){
	if(net_->has_layer(layer_name))
		return true;
	return false;
}
bool classifier::check_blob(string blob_name){
	if(net_->has_blob(blob_name))
		return true;
	return false;
}
int classifier::set_cont(int cont) {
	cont_ = cont;
	return 0;
}
vector<float> classifier::calc(vector<Mat> imgs) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	Blob<float>* output_layer = net_->output_blobs()[0];
	Blob<float>* input_cont;
	cout<<input_layer->shape_string()<<endl;
	cout<<"imgs.size()"<<imgs.size()<<endl;
	if(mode_ == "lstm") {
		input_layer->Reshape(imgs.size(), 3, 224, 224);
		input_cont = net_->input_blobs()[1];
		cout<<input_cont->shape_string()<<endl;
		vector<int> __shape__;
		__shape__.push_back(imgs.size());
		__shape__.push_back(1);
		input_cont->Reshape(__shape__);
	}
	net_->Reshape();
	cout<<"Net_Reshape!!!"<<endl;
	float* input_data = input_layer->mutable_cpu_data();
	cout<<"input.num() = "<<input_layer->num()<<endl;
	if(mode_ == "lstm") {
		float* input_cont_data = input_cont->mutable_cpu_data();
		float* cont_data = new float[imgs.size()];
		for(int i = 0; i < imgs.size(); i++) {
			if(i == 0)
				cont_data[i] = 0;
			else
				cont_data[i] = 1;
		}
		input_cont_data = cont_data;
	}

	vector<Mat> input_channels;
	for (int j = 0; j < input_layer->num()*input_layer->channels(); j++) {
		Mat channel(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += input_layer->height() * input_layer->width();
	}
	vector<Mat> sample_resized;
	Mat sample_temp;
	for(int j = 0; j < imgs.size(); j++) {
		if (imgs[j].size() != input_geometry_)
			resize(imgs[j], sample_temp, input_geometry_);
		else
			sample_temp = imgs[j];
		sample_temp.convertTo(sample_temp, CV_32FC3);
		if(set_mean_)
			subtract(sample_temp, mean_, sample_temp);
		if(set_regular_)
			sample_temp = sample_temp.mul(0.0078125);
		sample_resized.push_back(sample_temp);
	}
	Mat Merge_all;
	cv::merge(sample_resized, Merge_all);

	split(Merge_all, input_channels);

	net_->Forward();
	
	const vector<Blob<float>* >& result = net_->output_blobs();

	const float* begin;
	const float* end;
	cout<<"result.size = "<<result[0]->num()<<endl;
	for(int j = 0; j < result.size(); j++) {
		const string& output_name = 
			net_->blob_names()[net_->output_blob_indices()[j]];
		if(output_name == "lstm_Y_flatten"){
			begin = result[j]->cpu_data();
			end = begin + result[j]->channels() * result[j]->num();
			vector<float>output = vector<float>(begin, begin + output_layer->channels());
			
			for(int l = 0; l < 8 * 128; l++){
				if(l % 128 == 0)
					cout<<endl<<endl;
				cout<<begin[l]<<",";
			}
		}
	}

	vector<float>output = vector<float>(begin, begin + output_layer->channels());
	return output;
}
classifier::classifier(string model_file,
                 string trained_file,
                 bool set_mean,
                 bool set_regular,
		 string mode,
		 int cont){
	cont_ = cont;
	mode_ = mode;
	set_mean_ = set_mean;
	set_regular_ = set_regular;
	
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);
	
	Blob<float>* input_layer = net_->input_blobs()[0];
	
	num_channels_ = input_layer->channels();
	
	input_geometry_ = Size(input_layer->width(), input_layer->height());
	
	vector<Mat> channels;
	for(int j = 0; j < num_channels_; j++) {
	      Mat channel(224,224,CV_32FC1,127.5);
	      channels.push_back(channel);
	}
	Mat mean;
	merge(channels, mean);
	Scalar channel_mean = cv::mean(mean);
	mean_ = Mat(input_geometry_, mean.type(), channel_mean);
	
}

int main(int argc, char** argv) {
	
	if (argc < 2) {
		cerr<<"missing videopath"<<endl;
		return 1;
	}
	
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(0);

	string videopath = argv[1];
	ifstream file(videopath.c_str());
	string line;
	vector<string> imgs_path;
	while(getline(file, line)) {
		imgs_path.push_back(line);
	}
	
	vector<cv::Mat> imgs;
	cv::Mat img;
	int width = 224;
	int height = 224;
	for(int i = 0; i < imgs_path.size(); i++) {
		img = imread(imgs_path[i]);
		if(img.cols > width && img.rows > height) {
			if(img.cols*1.0/width > img.rows*1.0/height) {
				cv::resize(img, img, Size(width, (int)(img.rows * width * 1.0 / img.cols)));
				copyMakeBorder(img, img, (height - img.rows) / 2, height - img.rows - (height - img.rows) / 2, 0, 0, BORDER_REPLICATE, Scalar(255,255,255));
			}
			else {
				cv::resize(img, img, Size((int)(img.cols * height * 1.0 / img.rows), height));
				copyMakeBorder(img, img, 0, 0, (width - img.cols) / 2, width - img.cols - (width - img.cols) / 2, BORDER_REPLICATE, Scalar(255,255,255));
			}
		}
		else if(img.cols > width && img.rows <= height) {
			cv::resize(img, img, Size(width, (int)(img.rows * width * 1.0 / img.cols)));
			copyMakeBorder(img, img, (height - img.rows) / 2, height - img.rows - (height - img.rows) / 2, 0, 0, BORDER_REPLICATE, Scalar(255,255,255));
		}
		else if(img.cols <= width && img.rows > height) {
			cv::resize(img, img, Size((int)(img.cols * height * 1.0 / img.rows), height));
			copyMakeBorder(img, img, 0, 0, (width - img.cols) / 2, width - img.cols - (width - img.cols) / 2, BORDER_REPLICATE, Scalar(255,255,255));
	
		}
		else {
			if(width * 1.0 / img.cols < height * 1.0 / img.rows) {
				cv::resize(img, img, Size(width, (int)(img.rows * width * 1.0 / img.cols)));
				copyMakeBorder(img, img, (height - img.rows) / 2, height - img.rows - (height - img.rows) / 2, 0, 0, BORDER_REPLICATE, Scalar(255,255,255));
			}
			else {
				cv::resize(img, img, Size((int)(img.cols * height * 1.0 / img.rows), height));
				copyMakeBorder(img, img, 0, 0, (width - img.cols) / 2, width - img.cols - (width - img.cols) / 2, BORDER_REPLICATE, Scalar(255,255,255));
			}
	
		}
		imgs.push_back(img);
	}

	string model_lstm_file = "/home/zfy/tracking/my/appearance_deploy_lstm2.prototxt";
	string model_testvector_file = "/home/zfy/tracking/my/appearance_deploy_testvector.prototxt";
	string model_pred_file = "/home/zfy/tracking/my/appearance_deploy_pred.prototxt";
	string trained_file = "/home/zfy/tracking/my/lstm_gaussian_iter_2036.caffemodel";
	
	classifier model_lstm_net(model_lstm_file, trained_file, true, true, "lstm", 0);

	vector<float> test_vector;

	cout<<imgs.size()<<endl;
	vector<float> pred_vector;
	vector<float> lstm_vector;
	vector<Mat> tmp_img1;
	for(int i = 1; i < imgs.size(); i++) {
		tmp_img1.push_back(imgs[1]);
	}
	lstm_vector = model_lstm_net.calc(tmp_img1);
	
return 0;
}
