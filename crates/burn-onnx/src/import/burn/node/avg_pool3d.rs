use super::prelude::*;

impl NodeCodegen for onnx_ir::node::avg_pool3d::AveragePool3dNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.strides.to_tokens();
        let count_include_pad = self.config.count_include_pad;
        let ceil_mode = self.config.ceil_mode;

        let input_spatial = onnx_ir::node::padding::static_spatial_dims(&self.inputs[0].ty);
        let padding = crate::burn::codegen::resolve_auto_pad_3d(
            &self.config.auto_pad,
            &self.config.padding,
            input_spatial.as_deref(),
            &self.config.kernel_size,
            &self.config.strides,
            &self.config.dilation,
        );

        Some(Field::new(
            self.name.clone(),
            quote! {
                AvgPool3d
            },
            quote! {
                let #name = AvgPool3dConfig::new(#kernel_size)
                    .with_strides(#strides)
                    .with_padding(#padding)
                    .with_count_include_pad(#count_include_pad)
                    .with_ceil_mode(#ceil_mode)
                    .init();
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::pool::AvgPool3d");
        imports.register("burn::nn::pool::AvgPool3dConfig");
        imports.register("burn::nn::PaddingConfig3d");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::avg_pool3d::{AveragePool3dNode, AveragePool3dNodeBuilder, AvgPool3dConfig};
    use onnx_ir::padding::{AutoPad, PaddingConfig3d};

    fn create_avg_pool3d_node(name: &str) -> AveragePool3dNode {
        let config = AvgPool3dConfig::new(
            [2, 2, 2],
            [2, 2, 2],
            PaddingConfig3d::Valid,
            false,
            [1, 1, 1],
            false,
            AutoPad::NotSet,
        );

        AveragePool3dNodeBuilder::new(name)
            .input_tensor("input", 5, DType::F32)
            .output_tensor("output", 5, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_avg_pool3d_field_init() {
        let node = create_avg_pool3d_node("pool1");
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = AvgPool3dConfig::new([2, 2, 2])
            .with_strides([2, 2, 2])
            .with_padding(PaddingConfig3d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_avg_pool3d_forward() {
        let node = create_avg_pool3d_node("pool1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }
}
