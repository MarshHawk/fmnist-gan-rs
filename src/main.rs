use candle_core::{DType, Error, Device, Module, Tensor, Result, utils};
use hf_hub::{api::sync::Api, Repo, RepoType};
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;
use std::io::{self, BufReader, Read};

fn load_parquet(parquet: SerializedFileReader<std::fs::File>) -> Result<(Tensor, Tensor)> {
    let samples = parquet.metadata().file_metadata().num_rows() as usize;
    let mut buffer_images: Vec<u8> = Vec::with_capacity(samples * 784);
    let mut buffer_labels: Vec<u8> = Vec::with_capacity(samples);
    for row in parquet.into_iter().flatten() {
        for (_name, field) in row.get_column_iter() {
            if let parquet::record::Field::Group(subrow) = field {
                for (_name, field) in subrow.get_column_iter() {
                    if let parquet::record::Field::Bytes(value) = field {
                        let image = image::load_from_memory(value.data()).unwrap();
                        buffer_images.extend(image.to_luma8().as_raw());
                    }
                }
            } else if let parquet::record::Field::Long(label) = field {
                buffer_labels.push(*label as u8);
            }
        }
    }
    let images = (Tensor::from_vec(buffer_images, (samples, 784), &Device::Cpu)?
        .to_dtype(DType::F32)?
        / 255.)?;
    let labels = Tensor::from_vec(buffer_labels, (samples,), &Device::Cpu)?;
    Ok((images, labels))
}

pub struct Dataset {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
    pub labels: usize,
}

fn load() -> Result<Dataset> {
    let api = Api::new().map_err(|e| Error::Msg(format!("Api error: {e}")))?;
    let dataset_id = "fashion_mnist".to_string();
    let repo = Repo::with_revision(
        dataset_id,
        RepoType::Dataset,
        "refs/convert/parquet".to_string(),
    );

    let repo = api.repo(repo);
    let test_parquet_filename = repo
        .get("fashion_mnist/test/0000.parquet")
        .map_err(|e| Error::Msg(format!("Api error: {e}")))?;

        let train_parquet_filename = repo
        .get("fashion_mnist/train/0000.parquet")
        .map_err(|e| Error::Msg(format!("Api error: {e}")))?;
    let test_parquet = SerializedFileReader::new(std::fs::File::open(test_parquet_filename)?)
        .map_err(|e| Error::Msg(format!("Parquet error: {e}")))?;
    let train_parquet = SerializedFileReader::new(std::fs::File::open(train_parquet_filename)?)
        .map_err(|e| Error::Msg(format!("Parquet error: {e}")))?;
    let (test_images, test_labels) = load_parquet(test_parquet)?;
    let (train_images, train_labels) = load_parquet(train_parquet)?;

    Ok(Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
    })
}

fn main() -> std::result::Result<(), Error> {
    let dataset = load()?;

    Ok(())
}
