use std::cell::UnsafeCell;

use crate::Context;

#[derive(Debug, Clone)]
pub struct Expr<'a> {
    coefficent: &'a [isize],
    bias: isize,
}

impl<'a> Expr<'a> {
    pub fn new(ctx: &'a Context, coefficent: &[isize], bias: isize) -> Self {
        let coefficent = ctx.arena.alloc_slice_copy(coefficent);
        Self { coefficent, bias }
    }
    pub fn vectorize_into(&self, target: &mut Vec<isize>) {
        target.extend_from_slice(self.coefficent);
        target.push(self.bias);
    }
    pub fn affine_dim(&self) -> usize {
        self.coefficent.len() + 1
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_expr_new<'a>(
    ctx: *mut Context,
    coefficent: *const isize,
    len: usize,
    bias: isize,
) -> *mut Expr<'a> {
    let ctx = &*ctx;
    let coefficent = unsafe { std::slice::from_raw_parts(coefficent, len) };
    ctx.arena
        .alloc(UnsafeCell::new(Expr::new(ctx, coefficent, bias)))
        .get_mut()
}

/*
ssize_t *slap_expr_get_coefficients(slap_expr_t);
size_t slap_expr_get_length(slap_expr_t);
ssize_t slap_expr_get_bias(slap_expr_t);
*/

#[no_mangle]
pub unsafe extern "C" fn slap_expr_get_coefficients(expr: *const Expr) -> *const isize {
    (*expr).coefficent.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn slap_expr_get_length(expr: *const Expr) -> usize {
    (*expr).coefficent.len()
}

#[no_mangle]
pub unsafe extern "C" fn slap_expr_get_bias(expr: *const Expr) -> isize {
    (*expr).bias
}
